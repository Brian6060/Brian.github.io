#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion

# 保证脚本可直接从项目外部路径启动。
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


PROJECT_ROOT = Path(
    "/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0"
)
DEFAULT_TEST_OUT_ROOT = PROJECT_ROOT / "outputs_test_improved"
DEFAULT_GT_PROCESSED_ROOT = PROJECT_ROOT / "processed_unet_train_improved"
DEFAULT_GT_MANIFEST_NAME = "val_pairs.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs_eval_report_improved"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved Test2_0 predictions against GT and export reports.")
    parser.add_argument("--test-out-root", type=Path, default=DEFAULT_TEST_OUT_ROOT)
    parser.add_argument("--gt-processed-root", type=Path, default=DEFAULT_GT_PROCESSED_ROOT)
    parser.add_argument("--gt-manifest-name", type=str, default=DEFAULT_GT_MANIFEST_NAME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.7)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_tif_2d(path: Path) -> np.ndarray:
    try:
        arr = tifffile.imread(str(path))
    except ValueError as exc:
        if "imagecodecs" not in str(exc).lower():
            raise
        with Image.open(path) as img:
            arr = np.asarray(img)
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Only support 2D tif. Got {path} with shape {arr.shape}")
    return arr


def stem_from_record(record: Dict[str, Any]) -> str:
    stem = record.get("stem")
    if stem:
        return str(stem)

    seq = record.get("seq")
    frame = record.get("frame")
    if seq is not None and frame is not None:
        return f"{seq}_{frame}"

    image_path = record.get("image_path") or record.get("image_copy_tif") or record.get("image_tif")
    if image_path:
        return Path(str(image_path)).stem

    raise KeyError(f"Cannot infer stem from record: {record}")


def resolve_image_path(record: Dict[str, Any]) -> Path:
    for key in ("image_copy_tif", "image_tif", "image_path"):
        value = record.get(key)
        if value:
            return Path(str(value))
    raise KeyError(f"No image tif path found in record: {record}")


def resolve_gt_binary(record: Dict[str, Any]) -> np.ndarray:
    binary_path = record.get("binary_npy")
    if not binary_path:
        raise KeyError(f"No binary GT path found in record: {record}")
    gt = np.load(str(Path(str(binary_path))))
    gt = np.asarray(gt)
    if gt.ndim != 2:
        raise ValueError(f"GT binary mask must be 2D. Got shape {gt.shape} from {binary_path}")
    return (gt > 0).astype(np.uint8)


def compute_binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred = pred_mask.astype(bool, copy=False)
    gt = gt_mask.astype(bool, copy=False)

    tp = int(np.logical_and(pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    total = tp + tn + fp + fn

    def safe_div(num: float, den: float) -> float:
        return float(num / den) if den > 0 else 0.0

    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    pixel_acc = safe_div(tp + tn, total)
    pixel_error = 1.0 - pixel_acc

    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "pixel_acc": pixel_acc,
        "pixel_error": pixel_error,
    }


def to_uint8_vis(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Only support 2D visualization input. Got shape {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max - arr_min < 1e-8:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - arr_min) / (arr_max - arr_min)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def make_prob_heatmap(prob_map: np.ndarray) -> np.ndarray:
    prob = np.clip(prob_map.astype(np.float32, copy=False), 0.0, 1.0)
    r = np.clip(255.0 * np.minimum(1.0, prob * 1.6), 0.0, 255.0)
    g = np.clip(255.0 * np.sin(np.pi * prob), 0.0, 255.0)
    b = np.clip(255.0 * (1.0 - prob**0.7), 0.0, 255.0)
    heatmap = np.stack([r, g, b], axis=-1)
    return heatmap.astype(np.uint8)


def save_tif(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.asarray(array))


def build_index(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for record in records:
        stem = stem_from_record(record)
        if stem in index:
            raise ValueError(f"Duplicated stem found: {stem}")
        index[stem] = record
    return index


def load_overlay_font(font_size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    candidate_names = [
        "DejaVuSans.ttf",
        "Arial.ttf",
        "Helvetica.ttf",
    ]
    for name in candidate_names:
        try:
            return ImageFont.truetype(name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_metric_box(
    overlay_rgb: np.ndarray,
    metrics: Optional[Dict[str, float]],
    metric_names: Optional[List[str]] = None,
) -> np.ndarray:
    if not metrics:
        return overlay_rgb

    metric_names = metric_names or ["dice", "iou", "recall"]
    label_map = {
        "dice": "Dice",
        "iou": "IoU",
        "recall": "Recall",
    }
    lines = []
    for key in metric_names:
        if key not in metrics:
            continue
        label = label_map.get(key, key)
        lines.append(f"{label:<6} {metrics[key]:.4f}")
    if not lines:
        return overlay_rgb

    image = Image.fromarray(overlay_rgb, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    font_size = max(10, min(18, min(width, height) // 28))
    font = load_overlay_font(font_size)
    pad_x = max(8, width // 60)
    pad_y = max(6, height // 70)
    line_gap = max(2, font_size // 6)
    inner_x = max(8, font_size // 2)
    inner_y = max(6, font_size // 3)

    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_width = max(box[2] - box[0] for box in line_boxes)
    text_height = sum((box[3] - box[1]) for box in line_boxes) + line_gap * (len(lines) - 1)

    box_width = text_width + inner_x * 2
    box_height = text_height + inner_y * 2
    left = max(0, width - pad_x - box_width)
    top = max(0, height - pad_y - box_height)
    right = min(width, left + box_width)
    bottom = min(height, top + box_height)

    box_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    box_draw = ImageDraw.Draw(box_layer)
    box_draw.rounded_rectangle(
        [(left, top), (right, bottom)],
        radius=max(6, font_size // 2),
        fill=(18, 18, 18, 160),
    )
    image = Image.alpha_composite(image, box_layer)

    text_draw = ImageDraw.Draw(image)
    cursor_y = top + inner_y
    for line, bbox in zip(lines, line_boxes):
        line_height = bbox[3] - bbox[1]
        text_draw.text((left + inner_x, cursor_y), line, font=font, fill=(255, 255, 255, 255))
        cursor_y += line_height + line_gap

    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def make_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    metrics: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    base = to_uint8_vis(image)
    overlay = np.stack([base, base, base], axis=-1)

    mask = pred_mask.astype(bool, copy=False)
    contour = np.logical_and(mask, np.logical_not(binary_erosion(mask)))
    overlay[contour] = np.array([255, 0, 0], dtype=np.uint8)
    return draw_metric_box(overlay, metrics)


def ensure_same_shape(stem: str, arrays: Dict[str, np.ndarray]) -> None:
    shapes = {name: tuple(np.asarray(arr).shape) for name, arr in arrays.items()}
    uniq_shapes = set(shapes.values())
    if len(uniq_shapes) != 1:
        raise ValueError(f"Shape mismatch for {stem}: {shapes}")


def write_metrics_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No metric rows to write.")

    fieldnames = [
        "stem",
        "seq",
        "frame",
        "height",
        "width",
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "pixel_acc",
        "pixel_error",
        "tp",
        "tn",
        "fp",
        "fn",
        "image_path",
        "gt_binary_path",
        "pred_npy_path",
        "prob_npy_path",
        "package_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    args.test_out_root = args.test_out_root.expanduser().resolve()
    args.gt_processed_root = args.gt_processed_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if not (0.0 < args.threshold < 1.0):
        raise ValueError(f"threshold must be in (0, 1), got {args.threshold}")

    pred_npy_dir = args.test_out_root / "pred_masks_npy"
    prob_npy_dir = args.test_out_root / "prob_maps_npy"
    inference_manifest_path = args.test_out_root / "inference_manifest.json"
    gt_manifest_path = args.gt_processed_root / "manifests" / args.gt_manifest_name

    required_inputs = [
        args.test_out_root,
        pred_npy_dir,
        prob_npy_dir,
        inference_manifest_path,
        args.gt_processed_root,
        gt_manifest_path,
    ]
    missing_inputs = [str(path) for path in required_inputs if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError("Missing required input paths:\n" + "\n".join(missing_inputs))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    packaged_root = args.out_dir / "packaged_results"
    metrics_csv_path = args.out_dir / "metrics_per_image.csv"
    summary_path = args.out_dir / "summary.json"
    run_meta_path = args.out_dir / "run_meta.json"

    inference_records = load_json(inference_manifest_path)
    gt_records = load_json(gt_manifest_path)
    if not isinstance(inference_records, list) or not isinstance(gt_records, list):
        raise TypeError("Both inference manifest and GT manifest must be JSON lists.")

    gt_index = build_index(gt_records)
    per_image_rows: List[Dict[str, Any]] = []

    metric_names = [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "pixel_acc",
        "pixel_error",
    ]
    metric_sums = {name: 0.0 for name in metric_names}

    print("========== Evaluate2_0 Config ==========")
    print(json.dumps({
        "test_out_root": args.test_out_root,
        "gt_processed_root": args.gt_processed_root,
        "gt_manifest_name": args.gt_manifest_name,
        "out_dir": args.out_dir,
        "threshold": args.threshold,
        "num_inference_records": len(inference_records),
        "num_gt_records": len(gt_records),
    }, indent=2, ensure_ascii=False, default=json_default))
    print("========================================")

    t0 = time.time()

    for idx, infer_rec in enumerate(inference_records, start=1):
        stem = stem_from_record(infer_rec)
        if stem not in gt_index:
            raise KeyError(f"Stem {stem} from inference results is missing in GT manifest {gt_manifest_path}")

        gt_rec = gt_index[stem]
        image_path = resolve_image_path(gt_rec)
        pred_npy_path = Path(str(infer_rec.get("pred_npy") or pred_npy_dir / f"{stem}.npy"))
        prob_npy_path = Path(str(infer_rec.get("prob_npy") or prob_npy_dir / f"{stem}.npy"))
        gt_binary_path = Path(str(gt_rec.get("binary_npy")))

        if not prob_npy_path.exists():
            raise FileNotFoundError(f"Missing prob map npy for {stem}: {prob_npy_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image tif for {stem}: {image_path}")
        if not gt_binary_path.exists():
            raise FileNotFoundError(f"Missing GT binary npy for {stem}: {gt_binary_path}")

        image = read_tif_2d(image_path)
        prob_map = np.load(str(prob_npy_path)).astype(np.float32, copy=False)
        pred_mask_from_test = None
        if pred_npy_path.exists():
            pred_mask_from_test = (np.load(str(pred_npy_path)) > 0).astype(np.uint8)
        pred_mask = (prob_map >= args.threshold).astype(np.uint8)
        gt_mask = resolve_gt_binary(gt_rec)

        shape_inputs = {
            "image": image,
            "pred_mask": pred_mask,
            "prob_map": prob_map,
            "gt_mask": gt_mask,
        }
        if pred_mask_from_test is not None:
            shape_inputs["pred_mask_from_test"] = pred_mask_from_test
        ensure_same_shape(stem, shape_inputs)

        metrics = compute_binary_metrics(pred_mask, gt_mask)
        for name in metric_names:
            metric_sums[name] += metrics[name]

        package_dir = packaged_root / stem
        save_tif(package_dir / "image.tif", image)
        save_tif(package_dir / "gt_mask_vis.tif", gt_mask.astype(np.uint8) * 255)
        save_tif(package_dir / "pred_mask_vis.tif", pred_mask.astype(np.uint8) * 255)
        save_tif(package_dir / "prob_heatmap.tif", make_prob_heatmap(prob_map))
        save_tif(package_dir / "segmentation_overlay.tif", make_overlay(image, pred_mask, metrics=metrics))

        meta = {
            "stem": stem,
            "seq": gt_rec.get("seq"),
            "frame": gt_rec.get("frame"),
            "image_path": image_path,
            "gt_binary_path": gt_binary_path,
            "pred_npy_path": pred_npy_path,
            "prob_npy_path": prob_npy_path,
            "threshold": args.threshold,
            "pred_source": "prob_map_rethresholded",
            "shape_hw": [int(image.shape[0]), int(image.shape[1])],
            "metrics": metrics,
            "exported_files": [
                package_dir / "image.tif",
                package_dir / "gt_mask_vis.tif",
                package_dir / "pred_mask_vis.tif",
                package_dir / "prob_heatmap.tif",
                package_dir / "segmentation_overlay.tif",
                package_dir / "meta.json",
            ],
        }
        (package_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=json_default),
            encoding="utf-8",
        )

        row = {
            "stem": stem,
            "seq": gt_rec.get("seq"),
            "frame": gt_rec.get("frame"),
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "image_path": str(image_path),
            "gt_binary_path": str(gt_binary_path),
            "pred_npy_path": str(pred_npy_path),
            "prob_npy_path": str(prob_npy_path),
            "package_dir": str(package_dir),
            "threshold": float(args.threshold),
        }
        row.update(metrics)
        per_image_rows.append(row)

        print(f"[{idx:03d}/{len(inference_records):03d}] {stem} evaluated.")

    elapsed = time.time() - t0
    write_metrics_csv(metrics_csv_path, per_image_rows)

    num_images = len(per_image_rows)
    summary = {
        "num_images": num_images,
        "avg_dice": metric_sums["dice"] / num_images if num_images else 0.0,
        "avg_iou": metric_sums["iou"] / num_images if num_images else 0.0,
        "avg_precision": metric_sums["precision"] / num_images if num_images else 0.0,
        "avg_recall": metric_sums["recall"] / num_images if num_images else 0.0,
        "avg_specificity": metric_sums["specificity"] / num_images if num_images else 0.0,
        "avg_pixel_acc": metric_sums["pixel_acc"] / num_images if num_images else 0.0,
        "avg_pixel_error": metric_sums["pixel_error"] / num_images if num_images else 0.0,
        "test_out_root": args.test_out_root,
        "gt_processed_root": args.gt_processed_root,
        "gt_manifest_name": args.gt_manifest_name,
        "out_dir": args.out_dir,
        "threshold": args.threshold,
        "elapsed_sec": elapsed,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    run_meta = {
        "script": Path(__file__).resolve(),
        "test_out_root": args.test_out_root,
        "gt_processed_root": args.gt_processed_root,
        "gt_manifest_path": gt_manifest_path,
        "inference_manifest_path": inference_manifest_path,
        "pred_masks_npy_dir": pred_npy_dir,
        "prob_maps_npy_dir": prob_npy_dir,
        "out_dir": args.out_dir,
        "threshold": args.threshold,
        "packaged_results_dir": packaged_root,
        "metrics_per_image_csv": metrics_csv_path,
        "summary_json": summary_path,
        "num_images": num_images,
        "generated_at_unix": time.time(),
    }
    run_meta_path.write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    print("========== Evaluation Summary ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))
    print("========================================")
    print(f"Saved packaged results : {packaged_root}")
    print(f"Saved metrics csv      : {metrics_csv_path}")
    print(f"Saved summary          : {summary_path}")
    print(f"Saved run meta         : {run_meta_path}")


if __name__ == "__main__":
    main()
