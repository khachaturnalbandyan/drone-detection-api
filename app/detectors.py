"""Detector backends behind one small interface.

All backends implement ``predict_batch(images) -> list[list[Detection]]``.
The API and the batcher only depend on that method, so backends can be
swapped by config (and faked in tests) without touching anything else.
"""

import ast
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.config import Settings
from app.processing import letterbox, postprocess, to_input_tensor


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


class Detector(Protocol):
    name: str

    def predict_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Run detection on BGR uint8 images; one list of detections per image."""
        ...


class OnnxDetector:
    """YOLOv8 exported to ONNX, run with onnxruntime.

    Needs only numpy + opencv + onnxruntime at runtime (no PyTorch), which
    keeps the Docker image several times smaller than a torch-based one.
    """

    name = "onnx"

    def __init__(self, settings: Settings) -> None:
        import onnxruntime as ort  # imported lazily so other backends don't need it

        self.s = settings
        providers = [
            p
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()
        ]
        self.session = ort.InferenceSession(settings.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        # Ultralytics writes the class names into the ONNX metadata as a
        # Python-dict literal, e.g. "{0: 'drone'}".
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names: dict[int, str] = ast.literal_eval(meta.get("names", "{0: 'drone'}"))

    def predict_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        boxed, infos = zip(*(letterbox(img, self.s.imgsz) for img in images), strict=True)
        raw = self.session.run(None, {self.input_name: to_input_tensor(list(boxed))})[0]
        return [
            [
                Detection(*box, conf, cls, self.names.get(cls, str(cls)))
                for *box, conf, cls in postprocess(
                    raw[i],
                    info,
                    self.s.conf_threshold,
                    self.s.iou_threshold,
                    self.s.max_detections,
                )
            ]
            for i, info in enumerate(infos)
        ]


class TorchDetector:
    """The original Ultralytics/PyTorch model (``best.pt``). Reference backend."""

    name = "torch"

    def __init__(self, settings: Settings) -> None:
        from ultralytics import YOLO  # heavy import, only when this backend is used

        self.s = settings
        self.model = YOLO(settings.model_path)
        self.names: dict[int, str] = self.model.names

    def predict_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        results = self.model.predict(
            images,
            imgsz=self.s.imgsz,
            conf=self.s.conf_threshold,
            iou=self.s.iou_threshold,
            max_det=self.s.max_detections,
            verbose=False,
        )
        out = []
        for r in results:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            out.append(
                [
                    Detection(*map(float, b), float(c), int(k), self.names[int(k)])
                    for b, c, k in zip(xyxy, confs, classes, strict=True)
                ]
            )
        return out


class FakeDetector:
    """Deterministic stand-in used in tests and CI (no model file needed).

    Returns one 'drone' box covering the central quarter of each image and
    records the size of every batch it receives, so tests can assert that
    dynamic batching actually happened.
    """

    name = "fake"

    def __init__(self, settings: Settings | None = None, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.batch_sizes: list[int] = []

    def predict_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        import time

        self.batch_sizes.append(len(images))
        if self.delay_s:
            time.sleep(self.delay_s)
        out = []
        for img in images:
            h, w = img.shape[:2]
            out.append([Detection(w / 4, h / 4, 3 * w / 4, 3 * h / 4, 0.9, 0, "drone")])
        return out


def build_detector(settings: Settings) -> Detector:
    backends = {"onnx": OnnxDetector, "torch": TorchDetector, "fake": FakeDetector}
    return backends[settings.backend](settings)
