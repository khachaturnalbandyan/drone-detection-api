import asyncio

import numpy as np
import pytest

from app.batching import MicroBatcher
from app.detectors import FakeDetector

IMG = np.zeros((10, 10, 3), np.uint8)


async def test_concurrent_requests_are_batched():
    det = FakeDetector(delay_s=0.01)
    batcher = MicroBatcher(det, max_batch_size=8, max_wait_ms=50)
    await batcher.start()
    try:
        results = await asyncio.gather(*(batcher.submit(IMG) for _ in range(8)))
    finally:
        await batcher.stop()
    assert len(results) == 8 and all(len(r) == 1 for r in results)
    assert det.batch_sizes == [8]  # one model call for eight requests


async def test_batch_size_is_capped():
    det = FakeDetector()
    batcher = MicroBatcher(det, max_batch_size=4, max_wait_ms=50)
    await batcher.start()
    try:
        await asyncio.gather(*(batcher.submit(IMG) for _ in range(10)))
    finally:
        await batcher.stop()
    assert max(det.batch_sizes) <= 4 and sum(det.batch_sizes) == 10


async def test_single_request_is_not_held_longer_than_max_wait():
    batcher = MicroBatcher(FakeDetector(), max_batch_size=8, max_wait_ms=20)
    await batcher.start()
    try:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await batcher.submit(IMG)
        assert loop.time() - t0 < 0.5
    finally:
        await batcher.stop()


async def test_detector_error_reaches_caller_and_worker_survives():
    class Flaky(FakeDetector):
        calls = 0

        def predict_batch(self, images):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("boom")
            return super().predict_batch(images)

    batcher = MicroBatcher(Flaky(), max_batch_size=1, max_wait_ms=1)
    await batcher.start()
    try:
        with pytest.raises(RuntimeError, match="boom"):
            await batcher.submit(IMG)
        assert len(await batcher.submit(IMG)) == 1  # next request still works
    finally:
        await batcher.stop()


async def test_submit_before_start_fails():
    with pytest.raises(RuntimeError):
        await MicroBatcher(FakeDetector()).submit(IMG)
