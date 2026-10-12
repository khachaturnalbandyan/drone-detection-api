"""Export a YOLOv8 .pt checkpoint to ONNX with a dynamic batch dimension.

    python scripts/export_onnx.py --weights models/best.pt

Writes models/best.onnx next to the weights. Requires the dev dependencies
(ultralytics + torch); the API itself does not.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    # dynamic=True -> batch (and spatial) dims are symbolic, so one ONNX file
    # serves any batch size the micro-batcher produces.
    # simplify=True -> constant folding / graph cleanup with onnxslim.
    out = model.export(
        format="onnx", imgsz=args.imgsz, dynamic=True, simplify=True, opset=args.opset
    )
    print(f"Exported: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
