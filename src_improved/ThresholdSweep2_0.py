#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import tifffile

# 保证脚本可直接从项目外部路径启动。
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


PROJECT_ROOT = Path(
    "/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0"
)
DEFAULT_TEST_OUT_ROOT = PROJECT_ROOT / "outputs_val_infer_improved"
DEFAULT_GT_PROCESSED_ROOT = PROJECT_ROOT / "processed_unet_train_improved"
DEFAULT_GT_MANIFEST_NAME = "val_pairs.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs_threshold_sweep_improved"
DEFAULT_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep segmentation thresholds on saved val-set probability maps.")
    parser.add_argument("--test-out-root", type=Path, default=DEFAULT_TEST_OUT_ROOT)
    parser.add_argument("--gt-processed-root", type=Path, default=DEFAULT_GT_PROCESSED_ROOT)
    parser.add_argument("--gt-manifest-name", type=str, default=DEFAULT_GT_MANIFEST_NAME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_tif_2d(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
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

    return {
        "dice": safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": safe_div(tp, tp + fp + fn),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "pixel_acc": safe_div(tp + tn, total),
        "pixel_error": 1.0 - safe_div(tp + tn, total),
    }


def build_index(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for record in records:
        stem = stem_from_record(record)
        if stem in index:
            raise ValueError(f"Duplicated stem found: {stem}")
        index[stem] = record
    return index


def evaluate_threshold(dataset: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    metric_names = ["dice", "iou", "precision", "recall", "specificity", "pixel_acc", "pixel_error"]
    metric_sums = {name: 0.0 for name in metric_names}

    for sample in dataset:
        pred = (sample["prob_map"] >= threshold).astype(np.uint8)
        metrics = compute_binary_metrics(pred, sample["gt_mask"])
        for name in metric_names:
            metric_sums[name] += metrics[name]

    num_images = len(dataset)
    result = {
        "threshold": float(threshold),
        "num_images": num_images,
        "avg_dice": metric_sums["dice"] / num_images,
        "avg_iou": metric_sums["iou"] / num_images,
        "avg_precision": metric_sums["precision"] / num_images,
        "avg_recall": metric_sums["recall"] / num_images,
        "avg_specificity": metric_sums["specificity"] / num_images,
        "avg_pixel_acc": metric_sums["pixel_acc"] / num_images,
        "avg_pixel_error": metric_sums["pixel_error"] / num_images,
    }
    return result


def run_threshold_sweep(dataset: List[Dict[str, Any]], thresholds: List[float]) -> List[Dict[str, Any]]:
    results = []
    for threshold in thresholds:
        results.append(evaluate_threshold(dataset, threshold))
    return results


def save_results_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "threshold",
        "num_images",
        "avg_dice",
        "avg_iou",
        "avg_precision",
        "avg_recall",
        "avg_specificity",
        "avg_pixel_acc",
        "avg_pixel_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def choose_best(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    return max(rows, key=lambda row: (row[key], -abs(row["threshold"] - 0.50), -row["threshold"]))


def make_fine_thresholds(best_threshold: float) -> List[float]:
    start = max(0.05, best_threshold - 0.05)
    end = min(0.95, best_threshold + 0.05)
    values = np.arange(start, end + 1e-9, 0.01)
    rounded = sorted({round(float(v), 2) for v in values})
    return rounded


def print_table(title: str, rows: List[Dict[str, Any]]) -> None:
    print(title)
    print(
        f"{'thr':>5} {'dice':>9} {'iou':>9} {'prec':>9} {'recall':>9} {'spec':>9} {'pix_acc':>9} {'pix_err':>9}"
    )
    for row in rows:
        print(
            f"{row['threshold']:>5.2f} "
            f"{row['avg_dice']:>9.4f} "
            f"{row['avg_iou']:>9.4f} "
            f"{row['avg_precision']:>9.4f} "
            f"{row['avg_recall']:>9.4f} "
            f"{row['avg_specificity']:>9.4f} "
            f"{row['avg_pixel_acc']:>9.4f} "
            f"{row['avg_pixel_error']:>9.4f}"
        )


def main() -> None:
    args = parse_args()
    args.test_out_root = args.test_out_root.expanduser().resolve()
    args.gt_processed_root = args.gt_processed_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()

    prob_dir = args.test_out_root / "prob_maps_npy"
    inference_manifest_path = args.test_out_root / "inference_manifest.json"
    gt_manifest_path = args.gt_processed_root / "manifests" / args.gt_manifest_name

    required_paths = [
        args.test_out_root,
        prob_dir,
        inference_manifest_path,
        args.gt_processed_root,
        gt_manifest_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input paths:\n" + "\n".join(missing))

    inference_records = load_json(inference_manifest_path)
    gt_records = load_json(gt_manifest_path)
    if not isinstance(inference_records, list) or not isinstance(gt_records, list):
        raise TypeError("Both inference manifest and GT manifest must be JSON lists.")

    gt_index = build_index(gt_records)
    dataset: List[Dict[str, Any]] = []

    for infer_rec in inference_records:
        stem = stem_from_record(infer_rec)
        if stem not in gt_index:
            raise KeyError(f"Stem {stem} from inference results is missing in GT manifest.")

        gt_rec = gt_index[stem]
        prob_path = Path(str(infer_rec.get("prob_npy") or (prob_dir / f"{stem}.npy")))
        if not prob_path.exists():
            raise FileNotFoundError(f"Missing probability map for {stem}: {prob_path}")

        prob_map = np.load(str(prob_path)).astype(np.float32, copy=False)
        gt_mask = resolve_gt_binary(gt_rec)
        if prob_map.shape != gt_mask.shape:
            raise ValueError(f"Shape mismatch for {stem}: prob {prob_map.shape}, gt {gt_mask.shape}")

        dataset.append({
            "stem": stem,
            "prob_map": prob_map,
            "gt_mask": gt_mask,
            "prob_path": prob_path,
            "gt_binary_path": Path(str(gt_rec["binary_npy"])),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    coarse_rows = run_threshold_sweep(dataset, DEFAULT_THRESHOLDS)
    best_dice = choose_best(coarse_rows, "avg_dice")
    best_iou = choose_best(coarse_rows, "avg_iou")
    baseline_050 = next(row for row in coarse_rows if abs(row["threshold"] - 0.50) < 1e-9)

    print_table("===== Coarse Threshold Sweep =====", coarse_rows)
    print(
        f"Best avg_dice threshold: {best_dice['threshold']:.2f} "
        f"(avg_dice={best_dice['avg_dice']:.4f}, delta_vs_0.50={best_dice['avg_dice'] - baseline_050['avg_dice']:+.4f})"
    )
    print(
        f"Best avg_iou threshold : {best_iou['threshold']:.2f} "
        f"(avg_iou={best_iou['avg_iou']:.4f}, delta_vs_0.50={best_iou['avg_iou'] - baseline_050['avg_iou']:+.4f})"
    )

    save_results_csv(args.out_dir / "threshold_sweep.csv", coarse_rows)
    coarse_summary = {
        "num_images": len(dataset),
        "thresholds": DEFAULT_THRESHOLDS,
        "baseline_0.50": baseline_050,
        "best_by_dice": best_dice,
        "best_by_iou": best_iou,
    }
    save_json(args.out_dir / "threshold_sweep_summary.json", coarse_summary)
    save_json(args.out_dir / "best_by_dice.json", best_dice)
    save_json(args.out_dir / "best_by_iou.json", best_iou)

    best_threshold = best_dice["threshold"]
    if DEFAULT_THRESHOLDS[0] < best_threshold < DEFAULT_THRESHOLDS[-1]:
        fine_thresholds = make_fine_thresholds(best_threshold)
        fine_rows = run_threshold_sweep(dataset, fine_thresholds)
        fine_best_dice = choose_best(fine_rows, "avg_dice")
        fine_best_iou = choose_best(fine_rows, "avg_iou")

        print_table("===== Fine Threshold Sweep =====", fine_rows)
        print(
            f"Fine best avg_dice threshold: {fine_best_dice['threshold']:.2f} "
            f"(avg_dice={fine_best_dice['avg_dice']:.4f}, delta_vs_0.50={fine_best_dice['avg_dice'] - baseline_050['avg_dice']:+.4f})"
        )
        print(
            f"Fine best avg_iou threshold : {fine_best_iou['threshold']:.2f} "
            f"(avg_iou={fine_best_iou['avg_iou']:.4f}, delta_vs_0.50={fine_best_iou['avg_iou'] - baseline_050['avg_iou']:+.4f})"
        )

        save_results_csv(args.out_dir / "threshold_sweep_fine.csv", fine_rows)
        save_json(args.out_dir / "best_by_dice_fine.json", fine_best_dice)
        save_json(args.out_dir / "best_by_iou_fine.json", fine_best_iou)


if __name__ == "__main__":
    main()
