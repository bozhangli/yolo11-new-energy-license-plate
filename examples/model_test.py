from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "ultralytics" / "assets" / "bus.jpg"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    missing_dependency = exc.name
    YOLO = None
else:
    missing_dependency = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test an Ultralytics YOLO model with a local image, folder, video, or stream.")
    parser.add_argument("--model", default="yolo11n.pt", help="Model path or model name.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Inference source path, URL, camera index, or directory.")
    parser.add_argument("--target", help="Only keep one target class, using a class id like 5 or a class name like bus.")
    parser.add_argument("--device", default="cpu", help="Inference device, for example cpu, 0, or 0,1.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per image.")
    parser.add_argument("--save", action="store_true", help="Save rendered predictions to runs/.")
    parser.add_argument("--project", default="runs/model-test", help="Output project directory used when --save is set.")
    parser.add_argument("--name", default="predict", help="Output run name used when --save is set.")
    parser.add_argument("--verbose", action="store_true", help="Print Ultralytics inference logs.")
    return parser.parse_args()


def resolve_target_class(target: str | None, names: dict[int, str]) -> list[int] | None:
    if not target:
        return None
    if target.isdigit():
        class_id = int(target)
        if class_id not in names:
            raise ValueError(f"Unknown class id: {class_id}. Available ids: {sorted(names)}")
        return [class_id]

    normalized_target = target.strip().lower()
    for class_id, class_name in names.items():
        if class_name.lower() == normalized_target:
            return [class_id]
    available_names = ", ".join(names.values())
    raise ValueError(f"Unknown class name: {target}. Available names: {available_names}")


def summarize_result(index: int, result) -> None:
    source = getattr(result, "path", f"sample_{index}")
    boxes = len(result.boxes) if result.boxes is not None else 0
    masks = 0 if result.masks is None else len(result.masks)
    keypoints = 0 if result.keypoints is None else len(result.keypoints)
    obb = 0 if result.obb is None else len(result.obb)
    probs = None if result.probs is None else result.probs.top1
    speed = result.speed or {}

    print(f"[{index}] source: {source}")
    print(f"    boxes={boxes} masks={masks} keypoints={keypoints} obb={obb}")
    if probs is not None:
        top1_name = result.names.get(probs, str(probs))
        top1_conf = float(result.probs.top1conf)
        print(f"    top1={top1_name} conf={top1_conf:.4f}")
    summary = result.verbose().strip()
    if summary:
        print(f"    summary: {summary}")
    if speed:
        speed_text = ", ".join(f"{key}={value:.1f}ms" for key, value in speed.items())
        print(f"    speed: {speed_text}")


def save_result_info(index: int, result, save_dir: Path) -> Path:
    source = getattr(result, "path", f"sample_{index}")
    file_stem = Path(str(source)).stem or f"sample_{index}"
    json_path = save_dir / f"{file_stem}.json"
    json_path.write_text(result.to_json(), encoding="utf-8")
    return json_path


def main() -> int:
    args = parse_args()
    if YOLO is None:
        print(
            "Failed to import ultralytics runtime dependency: "
            f"{missing_dependency}. Install the project dependencies first, for example: pip install -e ."
        )
        return 1
    model = YOLO(args.model)
    try:
        target_classes = resolve_target_class(args.target, model.names)
    except ValueError as exc:
        print(exc)
        return 1
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        classes=target_classes,
        max_det=args.max_det,
        device=args.device,
        save=args.save,
        project=args.project,
        name=args.name,
        verbose=args.verbose,
    )

    print(f"model: {args.model}")
    print(f"source: {args.source}")
    if target_classes is not None:
        print(f"target: {model.names[target_classes[0]]} ({target_classes[0]})")
    print(f"results: {len(results)}")
    save_dir = Path(results[0].save_dir) if args.save and results else Path(args.project) / args.name
    for index, result in enumerate(results, start=1):
        summarize_result(index, result)
        if args.save:
            json_path = save_result_info(index, result, save_dir)
            print(f"    info_saved_to: {json_path}")

    if args.save:
        print(f"saved_to: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())