import numpy as np
import pytest

from app.processing import LetterboxInfo, letterbox, nms, postprocess, to_input_tensor


def test_letterbox_keeps_aspect_and_pads():
    img = np.zeros((480, 640, 3), np.uint8)  # landscape 4:3
    out, info = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert info.scale == pytest.approx(1.0)
    assert info.pad_x == 0 and info.pad_y == 80  # (640 - 480) / 2
    assert (out[:80] == 114).all() and (out[-80:] == 114).all()  # gray bars top/bottom


def test_to_input_tensor_shape_range_and_channel_order():
    img = np.zeros((4, 4, 3), np.uint8)
    img[..., 2] = 255  # pure red in BGR
    t = to_input_tensor([img, img])
    assert t.shape == (2, 3, 4, 4) and t.dtype == np.float32
    assert t[0, 0].max() == 1.0 and t[0, 2].max() == 0.0  # red ends up in channel 0 (RGB)


def test_nms_suppresses_overlaps_and_keeps_separate_boxes():
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],  # overlaps box 0 (IoU ~0.68)
            [50, 50, 60, 60],  # far away
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7])
    assert nms(boxes, scores, iou_threshold=0.5).tolist() == [0, 2]
    # With a very permissive threshold nothing is suppressed.
    assert nms(boxes, scores, iou_threshold=0.99).tolist() == [0, 1, 2]


def _raw_output(boxes_cxcywh, class_scores):
    """Build a fake YOLOv8 output tensor of shape (4 + nc, N)."""
    return np.hstack([np.asarray(boxes_cxcywh, float), np.asarray(class_scores, float)]).T


def test_postprocess_filters_maps_back_and_is_class_aware():
    info = LetterboxInfo(scale=0.5, pad_x=0, pad_y=80, orig_w=1280, orig_h=960)
    raw = _raw_output(
        [[100, 180, 40, 40], [102, 182, 40, 40], [102, 182, 40, 40], [300, 300, 20, 20]],
        [[0.9, 0.0], [0.8, 0.0], [0.0, 0.85], [0.1, 0.1]],  # last one below conf
    )
    dets = postprocess(raw, info, conf_threshold=0.4, iou_threshold=0.5, max_detections=10)

    # Box 1 is suppressed by box 0 (same class); box 2 survives (other class).
    assert [(round(d[4], 2), d[5]) for d in dets] == [(0.9, 0), (0.85, 1)]
    x1, y1, x2, y2 = dets[0][:4]
    # Network coords (80..120, 160..200) -> remove pad_y=80 -> divide by 0.5.
    assert (x1, y1, x2, y2) == pytest.approx((160, 160, 240, 240))


def test_postprocess_respects_max_detections_and_empty():
    info = LetterboxInfo(1.0, 0, 0, 640, 640)
    raw = _raw_output([[i * 50 + 20, 20, 10, 10] for i in range(5)], [[0.9]] * 5)
    assert len(postprocess(raw, info, 0.4, 0.5, max_detections=3)) == 3
    assert postprocess(raw, info, 0.95, 0.5, max_detections=3) == []
