from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train a YOLO11n detector for CCPD green license plates.")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained checkpoint to fine-tune.")
    parser.add_argument("--data", type=Path, default=root / "ccpd_green.yaml", help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--batch", type=int, default=6, help="Batch size tuned for an 8 GB class GPU.")
    parser.add_argument("--device", default="0", help="Training device, e.g. 0, 0,1 or cpu.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers. Use 0 on Windows when pagefile is limited.")
    parser.add_argument("--project", default="runs/train", help="Output project directory.")
    parser.add_argument("--name", default="ccpd-green-yolo11n-8gb", help="Training run name.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation augmentation. Plates usually need low rotation.")
    parser.add_argument("--fliplr", type=float, default=0.0, help="Disable horizontal flip for license plate semantics.")
    parser.add_argument("--mosaic", type=float, default=0.3, help="Moderate mosaic augmentation.")
    parser.add_argument("--close-mosaic", type=int, default=15, help="Disable mosaic in final epochs.")
    parser.add_argument("--save-period", type=int, default=10, help="Checkpoint save period.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        degrees=args.degrees,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        save_period=args.save_period,
        pretrained=True,
        single_cls=True,
    )


if __name__ == "__main__":
    main()