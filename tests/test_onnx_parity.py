"""Optional: compare the ONNX backend to Ultralytics on real weights.

Skipped unless both files exist, so CI (which has no weights) stays light:
    DDA_TEST_PT=models/best.pt DDA_TEST_ONNX=models/best.onnx pytest tests/test_onnx_parity.py
"""

import os
from pathlib import Path

import cv2
import pytest

PT = os.getenv("DDA_TEST_PT", "models/best.pt")
ONNX = os.getenv("DDA_TEST_ONNX", "models/best.onnx")

pytestmark = pytest.mark.skipif(
    not (Path(PT).exists() and Path(ONNX).exists()), reason="model weights not available"
)


def test_onnx_matches_torch():
    ultralytics = pytest.importorskip("ultralytics")
    from app.config import Settings
    from app.detectors import OnnxDetector, TorchDetector
    from scripts.benchmark import parity

    assets = Path(ultralytics.__file__).parent / "assets"
    images = [cv2.imread(str(p)) for p in sorted(assets.glob("*.jpg"))]
    common = dict(conf_threshold=0.25, max_detections=100)
    t = TorchDetector(Settings(backend="torch", model_path=PT, **common)).predict_batch(images)
    o = OnnxDetector(Settings(backend="onnx", model_path=ONNX, **common)).predict_batch(images)

    matched, total, mean_iou = parity(t, o)
    assert matched == total
    assert mean_iou > 0.98
