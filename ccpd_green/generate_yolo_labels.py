from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate YOLO labels for CCPD images whose annotations are embedded in filenames."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dataset root containing train/ and val/ directories. Defaults to the script directory.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Class ID to write into YOLO labels. Defaults to 0.",
    )
    return parser.parse_args()


def parse_bbox_from_name(image_path: Path) -> tuple[int, int, int, int]:
    parts = image_path.stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"unexpected filename format: {image_path.name}")

    left_top, right_bottom = parts[2].split("_")
    x1_str, y1_str = left_top.split("&")
    x2_str, y2_str = right_bottom.split("&")
    x1, y1, x2, y2 = map(int, (x1_str, y1_str, x2_str, y2_str))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid bbox coordinates: {image_path.name}")

    return x1, y1, x2, y2


def to_yolo_bbox(image_path: Path) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = parse_bbox_from_name(image_path)

    with Image.open(image_path) as image:
        width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {image_path.name}")

    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return x_center, y_center, box_width, box_height


def iter_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def write_split_labels(root: Path, split: str, class_id: int) -> tuple[int, list[str]]:
    image_dir = root / split
    label_dir = root / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    written = 0

    for image_path in iter_images(image_dir):
        try:
            x_center, y_center, box_width, box_height = to_yolo_bbox(image_path)
        except Exception as exc:
            errors.append(f"{image_path.name}: {exc}")
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        label_path.write_text(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n",
            encoding="utf-8",
        )
        written += 1

    return written, errors


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    missing = [split for split in ("train", "val") if not (root / split).is_dir()]
    if missing:
        raise SystemExit(f"missing required directories under {root}: {', '.join(missing)}")

    total_written = 0
    total_errors: list[str] = []

    for split in ("train", "val"):
        written, errors = write_split_labels(root, split, args.class_id)
        total_written += written
        total_errors.extend(errors)
        print(f"{split}: wrote {written} label files to {root / 'labels' / split}")

    if total_errors:
        print(f"completed with {len(total_errors)} errors")
        for message in total_errors[:20]:
            print(f"  - {message}")
        if len(total_errors) > 20:
            print(f"  - ... {len(total_errors) - 20} more")
        return 1

    print(f"done: wrote {total_written} label files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())