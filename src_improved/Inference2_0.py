#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tifffile
from scipy.ndimage import binary_erosion, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package final inference results and export per-image statistics.")
    parser.add_argument("--test-out-root", type=Path, required=True, help="Root produced by Test2_0.py")
    parser.add_argument("--processed-root", type=Path, required=True, help="Processed root for this inference set")
    parser.add_argument("--processed-manifest-name", type=str, default="test_images.json")
    parser.add_argument("--gt-processed-root", type=Path, default=None)
    parser.add_argument("--gt-manifest-name", type=str, default="val_pairs.json")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_tif_2d(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Only support 2D tif. Got {path} with shape {arr.shape}")
    return arr


def stem_from_record(rec: Dict) -> str:
    return f'{rec["seq"]}_{rec["frame"]}'


def resolve_image_path(rec: Dict) -> Path:
    for key in ["image_copy_tif", "image_tif"]:
        if key in rec:
            return Path(rec[key])
    raise KeyError("No image tif path found in record")


def resolve_gt_binary(rec: Dict) -> np.ndarray:
    if "binary_npy" in rec:
        return np.load(str(rec["binary_npy"])).astype(np.uint8)
    if "seg_tif" in rec:
        seg = read_tif_2d(Path(rec["seg_tif"]))
        return (seg > 0).astype(np.uint8)
    raise KeyError("No GT found in record.")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def save_tif(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr)


def build_index(records: List[Dict]) -> Dict[str, Dict]:
    return {stem_from_record(rec): rec for rec in records}


def to_uint8_vis(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8 and arr.min() >= 0 and arr.max() <= 255:
        return arr
    arr = arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip(np.round((arr - mn) / (mx - mn) * 255.0), 0, 255).astype(np.uint8)


def make_overlay(image: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = to_uint8_vis(image)
    rgb = np.stack([base, base, base], axis=-1)
    mask = pred.astype(bool)
    contour = np.logical_and(mask, np.logical_not(binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)))
    rgb[contour] = np.array([255, 0, 0], dtype=np.uint8)
    return rgb


def compute_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    total = max(tp + tn + fp + fn, 1)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pixel_acc": float((tp + tn) / total),
        "pixel_error": float((fp + fn) / total),
        "iou": float(tp / max(tp + fp + fn, 1)),
        "dice": float((2 * tp) / max(2 * tp + fp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
    }


def compute_prediction_side_stats(pred: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(np.uint8)
    prob = prob.astype(np.float32)
    total_pixels = int(pred.size)
    _, component_count = label(pred > 0)
    return {
        "positive_pixels": int(pred.sum()),
        "foreground_ratio": float(pred.sum() / max(total_pixels, 1)),
        "mean_prob": float(prob.mean()),
        "std_prob": float(prob.std()),
        "min_prob": float(prob.min()),
        "max_prob": float(prob.max()),
        "uncertain_ratio_40_60": float(np.logical_and(prob >= 0.4, prob <= 0.6).sum() / max(total_pixels, 1)),
        "confident_fg_ratio_p_ge_0_9": float((prob >= 0.9).sum() / max(total_pixels, 1)),
        "confident_bg_ratio_p_le_0_1": float((prob <= 0.1).sum() / max(total_pixels, 1)),
        "component_count": int(component_count),
    }


def avg_from_rows(key: str, rows: List[Dict]) -> float:
    return float(sum(r[key] for r in rows) / max(len(rows), 1))


def main() -> None:
    args = parse_args()
    args.test_out_root = args.test_out_root.expanduser().resolve()
    args.processed_root = args.processed_root.expanduser().resolve()
    args.gt_processed_root = args.gt_processed_root.expanduser().resolve() if args.gt_processed_root is not None else None
    args.out_dir = args.out_dir.expanduser().resolve() if args.out_dir is not None else (args.test_out_root / "final_inference_output")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred_npy_dir = args.test_out_root / "pred_masks_npy"
    pred_tif_dir = args.test_out_root / "pred_masks_tif"
    prob_npy_dir = args.test_out_root / "prob_maps_npy"
    ckpt_meta_path = args.test_out_root / "checkpoint_meta.json"
    inference_manifest_path = args.test_out_root / "inference_manifest.json"

    processed_manifest_path = args.processed_root / "manifests" / args.processed_manifest_name
    if not processed_manifest_path.exists():
        raise FileNotFoundError(f"Missing processed manifest: {processed_manifest_path}")
    processed_records = load_json(processed_manifest_path)
    processed_index = build_index(processed_records)

    gt_index: Dict[str, Dict] = {}
    if args.gt_processed_root is not None:
        gt_manifest_path = args.gt_processed_root / "manifests" / args.gt_manifest_name
        if gt_manifest_path.exists():
            gt_index = build_index(load_json(gt_manifest_path))

    if inference_manifest_path.exists():
        base_records = load_json(inference_manifest_path)
    else:
        base_records = []
        for rec in processed_records:
            stem = stem_from_record(rec)
            base_records.append({
                "seq": rec["seq"],
                "frame": rec["frame"],
                "image_path": str(resolve_image_path(rec)),
                "pred_npy": str(pred_npy_dir / f"{stem}.npy"),
                "pred_tif": str(pred_tif_dir / f"{stem}.tif"),
                "prob_npy": str(prob_npy_dir / f"{stem}.npy"),
            })

    ckpt_meta = load_json(ckpt_meta_path) if ckpt_meta_path.exists() else {}
    packaged_root = args.out_dir / "packaged_results"
    packaged_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    t0 = time.time()

    for idx, base in enumerate(base_records, start=1):
        seq = base["seq"]
        frame = base["frame"]
        stem = f"{seq}_{frame}"
        if stem not in processed_index:
            raise KeyError(f"Stem {stem} not found in processed manifest {processed_manifest_path}")

        proc_rec = processed_index[stem]
        image_path = resolve_image_path(proc_rec)
        pred_npy_path = pred_npy_dir / f"{stem}.npy"
        prob_npy_path = prob_npy_dir / f"{stem}.npy"
        pred_tif_path = pred_tif_dir / f"{stem}.tif"
        if not pred_npy_path.exists():
            raise FileNotFoundError(f"Missing predicted mask npy: {pred_npy_path}")
        if not prob_npy_path.exists():
            raise FileNotFoundError(f"Missing probability map npy: {prob_npy_path}")

        image = read_tif_2d(image_path)
        pred = np.load(str(pred_npy_path)).astype(np.uint8)
        prob = np.load(str(prob_npy_path)).astype(np.float32)
        if image.shape != pred.shape:
            raise ValueError(f"Prediction/image shape mismatch for {stem}: pred={pred.shape}, image={image.shape}")
        if image.shape != prob.shape:
            raise ValueError(f"Probability/image shape mismatch for {stem}: prob={prob.shape}, image={image.shape}")

        sample_dir = packaged_root / stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_dst = sample_dir / "image.tif"
        pred_vis_dst = sample_dir / "pred_mask_vis.tif"
        prob_vis_dst = sample_dir / "prob_heatmap.tif"
        overlay_dst = sample_dir / "segmentation_overlay.tif"

        copy_file(image_path, image_dst)
        save_tif(pred_vis_dst, (pred * 255).astype(np.uint8))
        save_tif(prob_vis_dst, np.clip(np.round(prob * 255.0), 0, 255).astype(np.uint8))
        save_tif(overlay_dst, make_overlay(image, pred))
        np.save(str(sample_dir / "pred_mask.npy"), pred.astype(np.uint8))
        np.save(str(sample_dir / "prob_map.npy"), prob.astype(np.float32))

        stats = compute_prediction_side_stats(pred, prob)
        has_gt = stem in gt_index
        gt_vis_dst: Optional[Path] = None
        metrics: Dict[str, float] = {}
        if has_gt:
            gt_bin = resolve_gt_binary(gt_index[stem]).astype(np.uint8)
            if gt_bin.shape != image.shape:
                raise ValueError(f"GT/image shape mismatch for {stem}: gt={gt_bin.shape}, image={image.shape}")
            gt_vis_dst = sample_dir / "gt_mask_vis.tif"
            save_tif(gt_vis_dst, (gt_bin * 255).astype(np.uint8))
            np.save(str(sample_dir / "gt_mask.npy"), gt_bin.astype(np.uint8))
            metrics = compute_binary_metrics(pred, gt_bin)
        else:
            (sample_dir / "GT_NOT_AVAILABLE.txt").write_text(
                "No ground-truth mask matched this image in the provided GT manifest.\n",
                encoding="utf-8",
            )

        meta = {
            "seq": seq,
            "frame": frame,
            "stem": stem,
            "image_path": str(image_path),
            "pred_mask_npy": str(pred_npy_path),
            "pred_mask_tif": str(pred_tif_path) if pred_tif_path.exists() else None,
            "prob_map_npy": str(prob_npy_path),
            "packaged_image": str(image_dst),
            "packaged_pred_mask_vis": str(pred_vis_dst),
            "packaged_prob_heatmap": str(prob_vis_dst),
            "packaged_segmentation_overlay": str(overlay_dst),
            "packaged_gt_mask_vis": str(gt_vis_dst) if gt_vis_dst is not None else None,
            "has_ground_truth": has_gt,
            **stats,
            **metrics,
        }
        (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

        row = {
            "seq": seq,
            "frame": frame,
            "stem": stem,
            "has_ground_truth": has_gt,
            "image_path": str(image_path),
            "sample_dir": str(sample_dir),
            **stats,
            **metrics,
        }
        rows.append(row)
        print(f"[{idx:03d}/{len(base_records):03d}] {stem} packaged")

    elapsed = time.time() - t0
    metrics_csv = args.out_dir / "metrics_per_image.csv"
    summary_json = args.out_dir / "summary.json"
    run_meta_json = args.out_dir / "run_meta.json"

    fieldnames = [
        "seq", "frame", "stem", "has_ground_truth", "image_path", "sample_dir",
        "positive_pixels", "foreground_ratio", "mean_prob", "std_prob", "min_prob", "max_prob",
        "uncertain_ratio_40_60", "confident_fg_ratio_p_ge_0_9", "confident_bg_ratio_p_le_0_1", "component_count",
        "tp", "tn", "fp", "fn", "pixel_acc", "pixel_error", "iou", "dice", "precision", "recall", "specificity",
    ]
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, None) for k in fieldnames})

    gt_rows = [r for r in rows if r.get("has_ground_truth")]
    summary = {
        "test_out_root": str(args.test_out_root),
        "processed_root": str(args.processed_root),
        "processed_manifest_name": args.processed_manifest_name,
        "gt_processed_root": str(args.gt_processed_root) if args.gt_processed_root is not None else None,
        "packaged_root": str(packaged_root),
        "num_images_total": len(rows),
        "num_images_with_gt": len(gt_rows),
        "elapsed_sec": elapsed,
        "checkpoint_meta": ckpt_meta,
        "avg_positive_pixels": avg_from_rows("positive_pixels", rows),
        "avg_foreground_ratio": avg_from_rows("foreground_ratio", rows),
        "avg_mean_prob": avg_from_rows("mean_prob", rows),
        "avg_std_prob": avg_from_rows("std_prob", rows),
        "avg_uncertain_ratio_40_60": avg_from_rows("uncertain_ratio_40_60", rows),
        "avg_confident_fg_ratio_p_ge_0_9": avg_from_rows("confident_fg_ratio_p_ge_0_9", rows),
        "avg_confident_bg_ratio_p_le_0_1": avg_from_rows("confident_bg_ratio_p_le_0_1", rows),
        "avg_component_count": avg_from_rows("component_count", rows),
    }
    if gt_rows:
        summary.update({
            "avg_pixel_acc": avg_from_rows("pixel_acc", gt_rows),
            "avg_pixel_error": avg_from_rows("pixel_error", gt_rows),
            "avg_iou": avg_from_rows("iou", gt_rows),
            "avg_dice": avg_from_rows("dice", gt_rows),
            "avg_precision": avg_from_rows("precision", gt_rows),
            "avg_recall": avg_from_rows("recall", gt_rows),
            "avg_specificity": avg_from_rows("specificity", gt_rows),
        })

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    run_meta_json.write_text(
        json.dumps({
            "args": vars(args),
            "metrics_csv": str(metrics_csv),
            "summary_json": str(summary_json),
            "packaged_root": str(packaged_root),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("========== Final Inference Summary ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print("============================================")


if __name__ == "__main__":
    main()
