"""Pre- and post-processing for YOLOv8, written with NumPy only.

The ONNX backend cannot rely on Ultralytics, so this module reimplements the
two pieces Ultralytics normally does for us:

1. ``letterbox``  - resize keeping aspect ratio, pad to a square canvas.
2. ``postprocess`` - decode the raw (4 + num_classes, N) output tensor,
   filter by confidence, run NMS and map boxes back to the original image.
"""

from dataclasses import dataclass

import cv2
import numpy as np

PAD_VALUE = 114  # gray padding, the value Ultralytics trains with


@dataclass(frozen=True)
class LetterboxInfo:
    """What we need to undo the letterbox transform on predicted boxes."""

    scale: float  # resize factor applied to the original image
    pad_x: float  # left padding in pixels (in network-input coordinates)
    pad_y: float  # top padding in pixels
    orig_w: int
    orig_h: int


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize ``image`` (H, W, 3 BGR uint8) into a ``size x size`` canvas.

    The longer side becomes ``size``; the shorter side is padded equally on
    both ends. Stretching instead would distort drone shapes the model never
    saw during training.
    """
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2
    # Split odd padding as floor/ceil so the canvas is exactly size x size.
    left, top = int(np.floor(pad_x)), int(np.floor(pad_y))
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, LetterboxInfo(scale, left, top, w, h)


def to_input_tensor(images: list[np.ndarray]) -> np.ndarray:
    """Stack letterboxed BGR uint8 images into a (B, 3, H, W) float32 RGB tensor in [0, 1]."""
    batch = np.stack(images)  # (B, H, W, 3)
    batch = batch[..., ::-1]  # BGR -> RGB
    batch = batch.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    return np.ascontiguousarray(batch, dtype=np.float32) / 255.0


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(cx, cy, w, h) -> (x1, y1, x2, y2)."""
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box (4,) and many boxes (N, 4), all in xyxy format."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression. Returns kept indices, best score first.

    O(N^2) in the worst case, but N is small here: candidates were already
    filtered by the confidence threshold, typically leaving tens of boxes.
    """
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        ious = box_iou_one_to_many(boxes[best], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def postprocess(
    raw: np.ndarray,
    info: LetterboxInfo,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> list[tuple[float, float, float, float, float, int]]:
    """Turn one image's raw YOLOv8 output into final detections.

    ``raw`` has shape (4 + num_classes, num_anchors): rows 0-3 are box
    cx, cy, w, h in network-input pixels, the rest are per-class scores
    (already sigmoid-ed; YOLOv8 has no separate objectness score).

    Returns ``[(x1, y1, x2, y2, confidence, class_id), ...]`` in original
    image pixels, sorted by confidence.
    """
    preds = raw.T  # (num_anchors, 4 + nc)
    class_scores = preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores[np.arange(len(preds)), class_ids]

    mask = confidences >= conf_threshold
    if not mask.any():
        return []
    boxes = xywh_to_xyxy(preds[mask, :4])
    confidences, class_ids = confidences[mask], class_ids[mask]

    # Class-aware NMS in one call: shift every class into its own region of
    # coordinate space so boxes of different classes can never overlap.
    offsets = class_ids[:, None].astype(np.float32) * 10_000.0
    keep = nms(boxes + offsets, confidences, iou_threshold)[:max_detections]

    # Undo letterbox: remove padding, then rescale, then clip to the image.
    boxes = boxes[keep]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - info.pad_x) / info.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - info.pad_y) / info.scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, info.orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, info.orig_h)

    return [
        (*map(float, box), float(conf), int(cls))
        for box, conf, cls in zip(boxes, confidences[keep], class_ids[keep], strict=True)
    ]
