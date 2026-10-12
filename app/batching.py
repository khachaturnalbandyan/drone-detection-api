"""Dynamic (micro-)batching for model inference.

Why: a GPU runs a batch of 8 images in much less than 8x the time of one
image. When many clients call the API at once, grouping their requests into
batches raises throughput without changing the API.

How it works:

    request 1 --submit()--+
    request 2 --submit()--+--> asyncio.Queue --> worker task --> detector.predict_batch()
    request 3 --submit()--+                         |
                                                    +--> sets each request's Future

* ``submit`` puts (image, future) on a queue and awaits the future.
* One background worker takes the first waiting item, then keeps collecting
  more until the batch is full or ``max_wait_ms`` has passed.
* The (blocking, CPU/GPU-bound) model call runs in a worker thread via
  ``asyncio.to_thread`` so the event loop keeps accepting requests meanwhile.

Trade-off: a lone request may wait up to ``max_wait_ms`` extra. With the
default of 10 ms that is small next to model latency.
"""

import asyncio
import contextlib
import time

import numpy as np

from app.detectors import Detection, Detector


class MicroBatcher:
    def __init__(self, detector: Detector, max_batch_size: int = 8, max_wait_ms: float = 10.0):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.detector = detector
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000
        self._queue: asyncio.Queue[tuple[np.ndarray, asyncio.Future]] = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="micro-batcher")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        # Fail anything still waiting instead of leaving callers hanging forever.
        while not self._queue.empty():
            _, fut = self._queue.get_nowait()
            if not fut.done():
                fut.set_exception(RuntimeError("batcher stopped"))

    async def submit(self, image: np.ndarray) -> list[Detection]:
        if self._worker is None:
            raise RuntimeError("batcher not started")
        fut: asyncio.Future[list[Detection]] = asyncio.get_running_loop().create_future()
        await self._queue.put((image, fut))
        return await fut

    async def _collect_batch(self) -> list[tuple[np.ndarray, asyncio.Future]]:
        batch = [await self._queue.get()]  # block until there is at least one request
        deadline = time.monotonic() + self.max_wait_s
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), remaining))
            except TimeoutError:
                break
        return batch

    async def _run(self) -> None:
        while True:
            batch = await self._collect_batch()
            # Drop requests whose client already went away (future cancelled).
            batch = [(img, fut) for img, fut in batch if not fut.cancelled()]
            if not batch:
                continue
            images = [img for img, _ in batch]
            try:
                results = await asyncio.to_thread(self.detector.predict_batch, images)
            except Exception as exc:  # one bad batch must not kill the worker
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            for (_, fut), result in zip(batch, results, strict=True):
                if not fut.done():
                    fut.set_result(result)
