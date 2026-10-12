"""Compare the PyTorch and ONNX backends: output parity, latency, throughput.

    python scripts/benchmark.py --pt models/best.pt --onnx models/best.onnx --images path/to/imgs

Prints a Markdown table that can be pasted into the README.
"""

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from app.config import Settings
from app.detectors import Detection, OnnxDetector, TorchDetector
from app.processing import box_iou_one_to_many


def load_images(folder: Path, limit: int) -> list[np.ndarray]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    images = [cv2.imread(str(p)) for p in paths[:limit]]
    if not images:
        raise SystemExit(f"No images found in {folder}")
    return images


def parity(a: list[list[Detection]], b: list[list[Detection]]) -> tuple[int, int, float]:
    """Match each torch box to the best same-class ONNX box.

    Returns (matched, total, mean IoU of matches). Torch and ONNX resize
    slightly differently, so we expect IoU close to, not exactly, 1.0.
    """
    matched, total, ious = 0, 0, []
    for dets_a, dets_b in zip(a, b, strict=True):
        for d in dets_a:
            total += 1
            cands = [x for x in dets_b if x.class_id == d.class_id]
            if not cands:
                continue
            boxes = np.array([[x.x1, x.y1, x.x2, x.y2] for x in cands])
            iou = box_iou_one_to_many(np.array([d.x1, d.y1, d.x2, d.y2]), boxes).max()
            if iou > 0.5:
                matched += 1
                ious.append(float(iou))
    return matched, total, statistics.fmean(ious) if ious else 0.0


def time_batches(detector, images, batch_size: int, repeats: int) -> tuple[float, float]:
    """Return (median ms per batch, images per second)."""
    batch = (images * batch_size)[:batch_size]
    detector.predict_batch(batch)  # warm-up (allocations, lazy init)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        detector.predict_batch(batch)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return med * 1000, batch_size / med


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pt", required=True)
    p.add_argument("--onnx", required=True)
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--batch-sizes", default="1,4,8")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-det", type=int, default=100)
    args = p.parse_args()

    images = load_images(args.images, limit=16)
    common = dict(conf_threshold=args.conf, max_detections=args.max_det)
    torch_det = TorchDetector(Settings(backend="torch", model_path=args.pt, **common))
    onnx_det = OnnxDetector(Settings(backend="onnx", model_path=args.onnx, **common))

    m, t, iou = parity(torch_det.predict_batch(images), onnx_det.predict_batch(images))
    print(f"Parity: {m}/{t} torch detections matched by ONNX, mean IoU {iou:.3f}\n")

    print("| backend | batch | ms / batch | images / s |")
    print("|---|---:|---:|---:|")
    for bs in map(int, args.batch_sizes.split(",")):
        for det in (torch_det, onnx_det):
            ms, ips = time_batches(det, images, bs, args.repeats)
            print(f"| {det.name} | {bs} | {ms:.1f} | {ips:.1f} |")


if __name__ == "__main__":
    main()
