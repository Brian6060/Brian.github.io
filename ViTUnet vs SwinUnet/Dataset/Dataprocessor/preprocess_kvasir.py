from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess the Kvasir-SEG dataset into a stable processed layout.")
    default_root = Path(__file__).resolve().parents[2]
    default_raw_root = default_root / "Dataset" / "raw" / "Kvasir-SEG"
    default_processed_root = default_root / "Dataset" / "processed" / "Kvasir-SEG"
    parser.add_argument("--raw_root", type=Path, default=default_raw_root, help="Raw Kvasir-SEG dataset root.")
    parser.add_argument("--processed_root", type=Path, default=default_processed_root, help="Processed output root.")
    parser.add_argument("--train_count", type=int, default=800, help="Number of training samples.")
    parser.add_argument("--val_count", type=int, default=100, help="Number of validation samples.")
    parser.add_argument("--test_count", type=int, default=100, help="Number of test samples.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    return parser.parse_args()


def load_bbox_index(json_path: Path) -> Dict[str, Any]:
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Failed to parse bbox JSON: {json_path}") from error


def summarize_numeric(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(array.min()), 6),
        "max": round(float(array.max()), 6),
        "mean": round(float(array.mean()), 6),
        "median": round(float(np.median(array)), 6),
        "std": round(float(array.std()), 6),
    }


def binarize_mask(mask_array: np.ndarray) -> np.ndarray:
    """Convert a noisy JPG mask into a clean binary mask."""

    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    return ((mask_array > 127).astype(np.uint8) * 255).astype(np.uint8)


def preprocess_dataset(
    raw_root: Path,
    processed_root: Path,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> Dict[str, Any]:
    images_dir = raw_root / "images"
    masks_dir = raw_root / "masks"
    bbox_json = raw_root / "kavsir_bboxes.json"

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Expected raw dataset folders under {raw_root}")

    processed_images_dir = processed_root / "images"
    processed_masks_dir = processed_root / "masks"
    processed_splits_dir = processed_root / "splits"
    processed_meta_dir = processed_root / "meta"
    for directory in (processed_images_dir, processed_masks_dir, processed_splits_dir, processed_meta_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_files = sorted([path for path in images_dir.iterdir() if path.is_file()])
    mask_files = sorted([path for path in masks_dir.iterdir() if path.is_file()])

    image_map = {path.stem: path for path in image_files}
    mask_map = {path.stem: path for path in mask_files}

    missing_images = sorted(set(mask_map.keys()) - set(image_map.keys()))
    missing_masks = sorted(set(image_map.keys()) - set(mask_map.keys()))
    common_stems = sorted(set(image_map.keys()) & set(mask_map.keys()))

    if len(common_stems) != (train_count + val_count + test_count):
        raise ValueError(
            "Split counts do not match the number of matched image/mask pairs: "
            f"{len(common_stems)} matched vs {train_count + val_count + test_count} requested."
        )

    bbox_index = load_bbox_index(bbox_json)
    rng = np.random.default_rng(seed)
    shuffled_stems = np.array(common_stems, dtype=object)
    rng.shuffle(shuffled_stems)

    split_boundaries = {
        "train": shuffled_stems[:train_count].tolist(),
        "val": shuffled_stems[train_count : train_count + val_count].tolist(),
        "test": shuffled_stems[train_count + val_count : train_count + val_count + test_count].tolist(),
    }

    all_records: List[Dict[str, Any]] = []
    widths: List[float] = []
    heights: List[float] = []
    foreground_ratios: List[float] = []
    anomalies: List[Dict[str, Any]] = []
    bbox_meta: Dict[str, Any] = {}

    for split_name, stems in split_boundaries.items():
        split_records: List[Dict[str, Any]] = []
        for stem in stems:
            image_path = image_map[stem]
            mask_path = mask_map[stem]

            try:
                with Image.open(image_path) as image_handle:
                    image = image_handle.convert("RGB")
                    width, height = image.size
                    processed_image_path = processed_images_dir / image_path.name
                    if not processed_image_path.exists():
                        shutil.copy2(image_path, processed_image_path)
            except Exception as error:  # pragma: no cover - defensive logging
                anomalies.append({"stem": stem, "issue": f"image_read_error: {error}"})
                continue

            try:
                with Image.open(mask_path) as mask_handle:
                    raw_mask = np.asarray(mask_handle.convert("L"), dtype=np.uint8)
            except Exception as error:  # pragma: no cover - defensive logging
                anomalies.append({"stem": stem, "issue": f"mask_read_error: {error}"})
                continue

            binary_mask = binarize_mask(raw_mask)
            if raw_mask.shape[:2] != (height, width):
                anomalies.append(
                    {
                        "stem": stem,
                        "issue": "size_mismatch",
                        "image_size": [width, height],
                        "mask_size": [int(raw_mask.shape[1]), int(raw_mask.shape[0])],
                    }
                )

            processed_mask_path = processed_masks_dir / f"{stem}.png"
            Image.fromarray(binary_mask).save(processed_mask_path)

            foreground_ratio = float((binary_mask > 0).sum() / max(binary_mask.size, 1))
            widths.append(float(width))
            heights.append(float(height))
            foreground_ratios.append(foreground_ratio)

            bbox_entry = bbox_index.get(stem)
            bbox_meta[stem] = bbox_entry if bbox_entry is not None else {}

            record = {
                "filename": image_path.name,
                "stem": stem,
                "split": split_name,
                "image_path": str(processed_image_path.resolve()),
                "mask_path": str(processed_mask_path.resolve()),
                "original_width": width,
                "original_height": height,
                "foreground_ratio": round(foreground_ratio, 6),
                "bbox_available": bool(bbox_entry),
                "bbox_count": len(bbox_entry.get("bbox", [])) if isinstance(bbox_entry, dict) else 0,
            }
            split_records.append(record)
            all_records.append(record)

        split_df = pd.DataFrame(split_records)
        split_df.sort_values("filename").to_csv(processed_splits_dir / f"{split_name}.csv", index=False)

    summary = {
        "dataset_name": "Kvasir-SEG",
        "seed": seed,
        "raw_root": str(raw_root.resolve()),
        "processed_root": str(processed_root.resolve()),
        "total_samples": len(all_records),
        "split_counts": {split_name: len(stems) for split_name, stems in split_boundaries.items()},
        "image_resolution_stats": {
            "width": summarize_numeric(widths),
            "height": summarize_numeric(heights),
        },
        "mask_foreground_ratio_stats": summarize_numeric(foreground_ratios),
        "bbox_summary": {
            "source_json": str(bbox_json.resolve()) if bbox_json.exists() else None,
            "matched_entries": int(sum(1 for stem in common_stems if stem in bbox_index)),
            "total_bbox_entries": int(len(bbox_index)),
        },
        "anomalies": {
            "missing_images": missing_images,
            "missing_masks": missing_masks,
            "other_issues": anomalies,
        },
    }

    (processed_meta_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (processed_meta_dir / "bboxes_index.json").write_text(json.dumps(bbox_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(all_records).sort_values(["split", "filename"]).to_csv(processed_meta_dir / "samples_manifest.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    summary = preprocess_dataset(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
