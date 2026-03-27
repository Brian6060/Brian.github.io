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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package final inference results and export per-image statistics.")
    parser.add_argument(
        "--test-out-root",
        type=Path,
        default=Path("/Users/dby051225/Desktop/VIS/U-Net/U-Net ER2.0/outputs_test_formal_A_best"),
        help="Root produced by Test.py",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/Users/dby051225/Desktop/VIS/U-Net/U-Net ER2.0/processed_unet_test"),
        help="Processed root for this inference set",
    )
    parser.add_argument(
        "--processed-manifest-name",
        type=str,
        default="test_images.json",
        help="Manifest under processed_root/manifests/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <test-out-root>/final_inference_output",
    )
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


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def save_tif(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr)


def build_index(records: List[Dict]) -> Dict[str, Dict]:
    return {stem_from_record(rec): rec for rec in records}


def count_components(binary: np.ndarray) -> int:
    # 简单 8 邻域连通域统计，不依赖额外库
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    count = 0
    ys, xs = np.where(binary)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx]:
            continue
        count += 1
        stack = [(int(sy), int(sx))]
        visited[sy, sx] = True
        while stack:
            y, x = stack.pop()
            for ny in range(max(0, y - 1), min(h, y + 2)):
                for nx in range(max(0, x - 1), min(w, x + 2)):
                    if not visited[ny, nx] and binary[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return count


def compute_pred_only_stats(image: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    # 没有 GT 时，不能算 IoU / Dice。这里只输出预测统计量。
    total_pixels = int(image.size)
    positive_pixels = int(pred.sum())
    foreground_ratio = positive_pixels / max(total_pixels, 1)
    mean_prob = float(prob.mean())
    std_prob = float(prob.std())
    min_prob = float(prob.min())
    max_prob = float(prob.max())
    uncertain_ratio_40_60 = float(((prob >= 0.4) & (prob <= 0.6)).sum() / max(total_pixels, 1))
    confident_fg_ratio = float((prob >= 0.9).sum() / max(total_pixels, 1))
    confident_bg_ratio = float((prob <= 0.1).sum() / max(total_pixels, 1))
    component_count = int(count_components(pred.astype(bool)))
    return {
        "total_pixels": total_pixels,
        "positive_pixels": positive_pixels,
        "foreground_ratio": float(foreground_ratio),
        "mean_prob": mean_prob,
        "std_prob": std_prob,
        "min_prob": min_prob,
        "max_prob": max_prob,
        "uncertain_ratio_40_60": uncertain_ratio_40_60,
        "confident_fg_ratio_p_ge_0_9": confident_fg_ratio,
        "confident_bg_ratio_p_le_0_1": confident_bg_ratio,
        "component_count": component_count,
    }


def main() -> None:
    args = parse_args()
    args.test_out_root = args.test_out_root.expanduser().resolve()
    args.processed_root = args.processed_root.expanduser().resolve()
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

        # 四件套
        image_dst = sample_dir / "image.tif"
        pred_vis_dst = sample_dir / "pred_mask_vis.tif"
        prob_vis_dst = sample_dir / "prob_heatmap.tif"
        gt_note_dst = sample_dir / "GT_NOT_AVAILABLE.txt"

        copy_file(image_path, image_dst)
        save_tif(pred_vis_dst, (pred * 255).astype(np.uint8))
        save_tif(prob_vis_dst, np.clip(np.round(prob * 255.0), 0, 255).astype(np.uint8))
        gt_note_dst.write_text("Official test set has no GT in current workflow, so true metrics like IoU/Dice cannot be computed here.\n", encoding="utf-8")

        # 原始数组也保存
        np.save(str(sample_dir / "pred_mask.npy"), pred.astype(np.uint8))
        np.save(str(sample_dir / "prob_map.npy"), prob.astype(np.float32))

        stats = compute_pred_only_stats(image=image, pred=pred, prob=prob)

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
            "has_ground_truth": False,
            **stats,
        }
        (sample_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        row = {
            "seq": seq,
            "frame": frame,
            "stem": stem,
            "image_path": str(image_path),
            "sample_dir": str(sample_dir),
            **stats,
        }
        rows.append(row)

        print(f"[{idx:03d}/{len(base_records):03d}] {stem} packaged | fg_ratio={stats['foreground_ratio']:.4f} mean_prob={stats['mean_prob']:.4f}")

    elapsed = time.time() - t0

    metrics_csv = args.out_dir / "metrics_per_image.csv"
    summary_json = args.out_dir / "summary.json"
    run_meta_json = args.out_dir / "run_meta.json"

    fieldnames = [
        "seq", "frame", "stem", "image_path", "sample_dir",
        "total_pixels", "positive_pixels", "foreground_ratio",
        "mean_prob", "std_prob", "min_prob", "max_prob",
        "uncertain_ratio_40_60",
        "confident_fg_ratio_p_ge_0_9",
        "confident_bg_ratio_p_le_0_1",
        "component_count",
    ]
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, None) for k in fieldnames})

    def avg(key: str) -> float:
        return float(sum(r[key] for r in rows) / max(len(rows), 1))

    summary = {
        "test_out_root": str(args.test_out_root),
        "processed_root": str(args.processed_root),
        "processed_manifest_name": args.processed_manifest_name,
        "packaged_root": str(packaged_root),
        "num_images_total": len(rows),
        "elapsed_sec": elapsed,
        "checkpoint_meta": ckpt_meta,
        "note": (
            "Current input is official test data without GT. Therefore true segmentation metrics such as IoU, Dice, Precision, Recall, Pixel Accuracy cannot be computed. "
            "This summary reports prediction-side statistics only, and packages each image with original image, predicted-mask visualization, and probability grayscale heatmap."
        ),
        "avg_foreground_ratio": avg("foreground_ratio"),
        "avg_mean_prob": avg("mean_prob"),
        "avg_std_prob": avg("std_prob"),
        "avg_uncertain_ratio_40_60": avg("uncertain_ratio_40_60"),
        "avg_confident_fg_ratio_p_ge_0_9": avg("confident_fg_ratio_p_ge_0_9"),
        "avg_confident_bg_ratio_p_le_0_1": avg("confident_bg_ratio_p_le_0_1"),
        "avg_component_count": avg("component_count"),
    }
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    run_meta = {
        "args": vars(args),
        "metrics_csv": str(metrics_csv),
        "summary_json": str(summary_json),
        "packaged_root": str(packaged_root),
    }
    run_meta_json.write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("========== Final Inference Summary ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print("============================================")
    print(f"Saved metrics CSV : {metrics_csv}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Packaged folders  : {packaged_root}")


if __name__ == "__main__":
    main()
