"""Send many concurrent requests to a running API and report throughput.

    python scripts/load_test.py --url http://localhost:8000 --image some.jpg -n 64 -c 16

Run it against the server with DDA_MAX_BATCH_SIZE=1 and then =8 to see what
dynamic batching buys you on your hardware.
"""

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import httpx


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--image", required=True)
    p.add_argument("-n", "--requests", type=int, default=64)
    p.add_argument("-c", "--concurrency", type=int, default=16)
    args = p.parse_args()

    data = Path(args.image).read_bytes()
    sem = asyncio.Semaphore(args.concurrency)  # at most `concurrency` requests in flight
    latencies: list[float] = []

    async with httpx.AsyncClient(base_url=args.url, timeout=120) as client:
        health = (await client.get("/health")).json()

        async def one() -> None:
            async with sem:
                t0 = time.perf_counter()
                r = await client.post("/v1/detect", files={"file": ("img.jpg", data, "image/jpeg")})
                r.raise_for_status()
                latencies.append(time.perf_counter() - t0)

        start = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(args.requests)))
        total = time.perf_counter() - start

    latencies.sort()
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    print(
        f"backend={health['backend']} max_batch={health['max_batch_size']} "
        f"requests={args.requests} concurrency={args.concurrency}\n"
        f"throughput={args.requests / total:.2f} req/s  "
        f"p50={statistics.median(latencies) * 1000:.0f} ms  p95={p95 * 1000:.0f} ms"
    )


if __name__ == "__main__":
    asyncio.run(main())
