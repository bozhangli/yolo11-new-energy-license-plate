import os
import re
import json
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from lprnet.model.LPRNet import build_lprnet
from lprnet.data.load_data import CHARS

YOLO_MODEL_PATH = r"D:\ProgramStudy\ultralytics_yolo11\runs\train\ccpd-green-yolo11n-8gb4\weights\best.pt"
LPR_MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "Final_LPRNet_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CCPD字符集
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
             "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
             "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z','O']
ADS = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
       '0','1','2','3','4','5','6','7','8','9','O']


def parse_ccpd_label(filename):
    m = re.search(r"-(\d+_\d+_\d+_\d+_\d+_\d+_\d+(?:_\d+)?)-", filename)
    if not m:
        return None
    parts = m.group(1).split("_")
    indices = [int(x) for x in parts]
    if len(indices) < 7:
        return None
    try:
        chars = [PROVINCES[indices[0]], ALPHABETS[indices[1]]]
        for i in indices[2:]:
            chars.append(ADS[i])
        return "".join(c for c in chars if c != "O")
    except (IndexError, ValueError):
        return None


def greedy_decode(prebs, chars):
    if isinstance(prebs, torch.Tensor):
        prebs = prebs.cpu().detach().numpy()
    if prebs.ndim == 3:
        prebs = prebs[0]

    preb_label = [np.argmax(prebs[:, j], axis=0) for j in range(prebs.shape[1])]

    no_repeat_blank_label = []
    if len(preb_label) > 0:
        pre_c = preb_label[0]
        if pre_c != len(chars) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label[1:]:
            if (pre_c == c) or (c == len(chars) - 1):
                if c == len(chars) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c

    return "".join(chars[i] if 0 <= i < len(chars) else "?" for i in no_repeat_blank_label) or "无法识别"


def _load_font(size=20):
    for path in ["simhei.ttf", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_cn_text(img, text, pos, color=(255, 0, 0), font_size=20):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _load_font(font_size)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def load_models(yolo_path=YOLO_MODEL_PATH, lpr_path=LPR_MODEL_PATH):
    print(f"加载 YOLO 模型: {yolo_path}")
    detector = YOLO(yolo_path)
    print(f"加载 LPRNet 模型: {lpr_path}")
    lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0.5)
    lprnet.load_state_dict(torch.load(lpr_path, map_location=DEVICE, weights_only=True))
    lprnet.to(DEVICE)
    lprnet.eval()
    return detector, lprnet


def recognize_plate(lprnet, plate_image):
    if plate_image is None or len(plate_image.shape) != 3:
        return "识别失败"
    img = cv2.resize(plate_image, (94, 24)).astype("float32")
    img = (img - 127.5) * 0.0078125
    img = img.transpose(2, 0, 1)[None]
    with torch.no_grad():
        prebs = lprnet(torch.from_numpy(img).to(DEVICE))
    return greedy_decode(prebs, CHARS)


def process_directory(input_dir, output_dir, detector, lprnet, show=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in exts:
        image_paths.extend(input_dir.rglob(ext))
    image_paths.sort()

    if not image_paths:
        print(f"在 {input_dir} 中未找到图片")
        return

    print(f"共找到 {len(image_paths)} 张图片")

    all_results = []
    for idx, img_path in enumerate(image_paths, 1):
        filename = img_path.name
        print(f"\n[{idx}/{len(image_paths)}] {filename}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  跳过: 无法读取")
            continue

        results = detector(img, verbose=False)
        plates = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = round(box.conf[0].item(), 4)
                crop = img[y1:y2, x1:x2]
                text = recognize_plate(lprnet, crop)

                gt = parse_ccpd_label(filename)
                match = (text == gt) if gt else None

                plates.append({
                    "coords": [x1, y1, x2, y2],
                    "confidence": conf,
                    "recognized": text,
                    "ground_truth": gt,
                    "match": match
                })

                status = "✓" if match else ("✗" if match is False else "?")
                gt_str = f" (GT: {gt})" if gt else ""
                print(f"  {status} {text}  (置信度: {conf:.2f}){gt_str}")

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                img = _draw_cn_text(img, f"{text} ({conf:.2f})", (x1, y1 - 24))

        result_item = {"file": filename, "path": str(img_path), "plates": plates, "plate_count": len(plates)}
        all_results.append(result_item)

        cv2.imwrite(str(output_dir / filename), img)

        if show:
            cv2.imshow(filename, img)
            key = cv2.waitKey(0)
            cv2.destroyWindow(filename)
            if key == 27:  # ESC 退出
                print("  用户中断")
                break

    # 统计
    total = sum(1 for r in all_results for p in r["plates"])
    correct = sum(1 for r in all_results for p in r["plates"] if p.get("match") is True)
    wrong = sum(1 for r in all_results for p in r["plates"] if p.get("match") is False)
    unknown = sum(1 for r in all_results for p in r["plates"] if p.get("match") is None)
    detected = sum(1 for r in all_results if r["plate_count"] > 0)

    print(f"\n{'='*50}")
    print(f"处理完成: {len(image_paths)} 张图片, {detected} 张检测到车牌, 共 {total} 个车牌")
    print(f"识别正确: {correct}, 识别错误: {wrong}, 无GT对照: {unknown}")
    if total > 0 and correct + wrong > 0:
        print(f"准确率: {correct / (correct + wrong) * 100:.1f}%")

    # 保存JSON结果
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {json_path}")


def process_single(image_path, output_dir, detector, lprnet, show=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    print(f"处理: {image_path}")

    results = detector(img)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            crop = img[y1:y2, x1:x2]
            text = recognize_plate(lprnet, crop)
            print(f"  识别结果: {text} (置信度: {conf:.2f})")

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img = _draw_cn_text(img, f"{text} ({conf:.2f})", (x1, y1 - 24))

    save_path = output_dir / Path(image_path).name
    cv2.imwrite(str(save_path), img)
    print(f"已保存: {save_path}")

    if show:
        cv2.imshow("车牌检测与识别", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLO11 + LPRNet 车牌检测识别")
    parser.add_argument("--input", help="输入图片或目录路径")
    parser.add_argument("--output", default=None, help="输出目录 (默认: runs/lpr_results)")
    parser.add_argument("--show", action="store_true", help="弹窗显示检测结果")
    args = parser.parse_args()

    detector, lprnet = load_models()

    if os.path.isdir(args.input):
        output_dir = args.output or os.path.join("runs", "lpr_results", os.path.basename(os.path.normpath(args.input)))
        process_directory(args.input, output_dir, detector, lprnet, show=args.show)
    else:
        output_dir = args.output or os.path.join("runs", "lpr_results", "single")
        process_single(args.input, output_dir, detector, lprnet, show=args.show)
