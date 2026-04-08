#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image


SOURCE_ROOT = Path(
    "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/Dataset 2D/raw/01_Multi-Atlas_Labeling"
)
IMAGE_DIR = SOURCE_ROOT / "img"
LABEL_DIR = SOURCE_ROOT / "label"
OUTPUT_ROOT = Path(
    "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/Dataset 2D/processed/BTCV_multiorgan_2d"
)

TARGET_SIZE = (256, 256)
SLICE_AXIS = 2
USE_HU_WINDOW = True
HU_WINDOW_MIN = -160.0
HU_WINDOW_MAX = 240.0
BACKGROUND_KEEP_RATIO = 0.1
RANDOM_SEED = 42
TASK_TYPE = "2d_multiclass_segmentation"
MAX_PREVIEWS_PER_CASE = 2

# BTCV commonly uses 13 foreground classes.
OFFICIAL_LABEL_MAP = {
    0: "background",
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior_vena_cava",
    10: "portal_and_splenic_vein",
    11: "pancreas",
    12: "right_adrenal_gland",
    13: "left_adrenal_gland",
}


def ensure_dirs() -> Dict[str, Path]:
    images_dir = OUTPUT_ROOT / "images"
    masks_dir = OUTPUT_ROOT / "masks"
    meta_dir = OUTPUT_ROOT / "meta"
    previews_dir = OUTPUT_ROOT / "previews"

    for directory in [OUTPUT_ROOT, images_dir, masks_dir, meta_dir, previews_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "images": images_dir,
        "masks": masks_dir,
        "meta": meta_dir,
        "previews": previews_dir,
    }


def _strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def _case_match_key(path: Path) -> str:
    stem = _strip_nii_suffix(path)
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return digits
    lower = stem.lower()
    for prefix in ["img", "image", "label", "seg", "mask"]:
        if lower.startswith(prefix):
            return lower[len(prefix):]
    return lower


def build_case_mapping(image_dir: Path, label_dir: Path) -> List[Dict[str, Path | str]]:
    image_files = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))]
    )
    label_files = sorted(
        [p for p in label_dir.iterdir() if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))]
    )

    image_map = {_case_match_key(path): path for path in image_files}
    label_map = {_case_match_key(path): path for path in label_files}

    shared_keys = sorted(set(image_map.keys()) & set(label_map.keys()))
    missing_image = sorted(set(label_map.keys()) - set(image_map.keys()))
    missing_label = sorted(set(image_map.keys()) - set(label_map.keys()))

    for key in missing_image:
        print(f"[Warning] Missing image for label key={key}, skip.")
    for key in missing_label:
        print(f"[Warning] Missing label for image key={key}, skip.")

    case_pairs: List[Dict[str, Path | str]] = []
    for key in shared_keys:
        image_path = image_map[key]
        label_path = label_map[key]
        case_id = _strip_nii_suffix(image_path)
        case_pairs.append(
            {
                "case_id": case_id,
                "image_path": image_path,
                "label_path": label_path,
            }
        )

    return case_pairs


def load_nifti_array(path: Path) -> np.ndarray:
    nii = nib.load(str(path))
    data = nii.get_fdata()
    return np.asarray(data)


def extract_slice(volume: np.ndarray, slice_index: int, slice_axis: int) -> np.ndarray:
    return np.take(volume, indices=slice_index, axis=slice_axis)


def resize_image_slice(image_slice: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(np.clip(image_slice * 255.0, 0, 255).astype(np.uint8), mode="L")
    resized = pil_image.resize(target_size, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def resize_mask_slice(mask_slice: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    pil_mask = Image.fromarray(mask_slice.astype(np.int32), mode="I")
    resized = pil_mask.resize(target_size, resample=Image.NEAREST)
    return np.asarray(resized, dtype=np.int64)


def normalize(image_slice: np.ndarray, use_hu_window: bool = True) -> np.ndarray:
    image_slice = image_slice.astype(np.float32, copy=False)
    if use_hu_window:
        image_slice = np.clip(image_slice, HU_WINDOW_MIN, HU_WINDOW_MAX)
        image_slice = (image_slice - HU_WINDOW_MIN) / (HU_WINDOW_MAX - HU_WINDOW_MIN)
        return np.clip(image_slice, 0.0, 1.0)

    min_val = float(image_slice.min())
    max_val = float(image_slice.max())
    if max_val <= min_val:
        return np.zeros_like(image_slice, dtype=np.float32)
    return (image_slice - min_val) / (max_val - min_val)


def choose_background_indices(
    background_indices: Sequence[int],
    background_keep_ratio: float,
    rng: random.Random,
) -> List[int]:
    if not background_indices:
        return []
    keep_count = int(round(len(background_indices) * background_keep_ratio))
    keep_count = max(1, keep_count) if background_keep_ratio > 0 else 0
    keep_count = min(keep_count, len(background_indices))
    if keep_count == 0:
        return []
    chosen = rng.sample(list(background_indices), keep_count)
    return sorted(chosen)


def save_overlay_preview(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    save_path: Path,
    case_id: str,
    slice_index: int,
) -> None:
    unique_classes = [int(v) for v in np.unique(mask_slice) if int(v) > 0]
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_slice, cmap="gray", vmin=0.0, vmax=1.0)

    for class_idx in unique_classes:
        binary_mask = (mask_slice == class_idx).astype(np.uint8)
        if binary_mask.sum() == 0:
            continue
        color = cmap((class_idx - 1) % 20)
        ax.contour(binary_mask, levels=[0.5], colors=[color], linewidths=1.0)

    ax.set_title(f"{case_id} | slice {slice_index}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_label_map(meta_dir: Path, detected_classes: Sequence[int]) -> Dict[int, str]:
    label_map: Dict[int, str] = {0: "background"}
    for class_idx in sorted(set(int(v) for v in detected_classes)):
        if class_idx == 0:
            continue
        label_map[class_idx] = OFFICIAL_LABEL_MAP.get(class_idx, f"organ_{class_idx}")

    with (meta_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2, ensure_ascii=False)
    return label_map


def write_preprocess_config(meta_dir: Path) -> None:
    config = {
        "source_root": str(SOURCE_ROOT),
        "image_dir": str(IMAGE_DIR),
        "label_dir": str(LABEL_DIR),
        "target_size": list(TARGET_SIZE),
        "slice_axis": SLICE_AXIS,
        "normalize_method": "hu_window" if USE_HU_WINDOW else "per_slice_minmax",
        "hu_window_min": HU_WINDOW_MIN,
        "hu_window_max": HU_WINDOW_MAX,
        "background_keep_ratio": BACKGROUND_KEEP_RATIO,
        "random_seed": RANDOM_SEED,
        "task_type": TASK_TYPE,
    }
    with (meta_dir / "preprocess_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def write_slice_csv(meta_dir: Path, slice_rows: Sequence[Dict[str, object]]) -> None:
    csv_path = meta_dir / "slices.csv"
    df = pd.DataFrame(slice_rows)
    df.to_csv(csv_path, index=False)


def split_cases(case_ids: Sequence[str]) -> Dict[str, List[str]]:
    rng = random.Random(RANDOM_SEED)
    shuffled = list(case_ids)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * 0.8)
    val_count = int(total * 0.1)
    test_count = total - train_count - val_count

    if total >= 3:
        val_count = max(1, val_count)
        test_count = max(1, test_count)
        train_count = total - val_count - test_count

    train_cases = shuffled[:train_count]
    val_cases = shuffled[train_count:train_count + val_count]
    test_cases = shuffled[train_count + val_count:]

    return {
        "train_cases": sorted(train_cases),
        "val_cases": sorted(val_cases),
        "test_cases": sorted(test_cases),
    }


def _safe_organ_classes(mask_slice: np.ndarray) -> List[int]:
    return [int(v) for v in np.unique(mask_slice) if int(v) > 0]


def process_case(
    case_info: Dict[str, Path | str],
    output_dirs: Dict[str, Path],
    rng: random.Random,
) -> Tuple[List[Dict[str, object]], Dict[str, int], List[int]]:
    case_id = str(case_info["case_id"])
    image_path = Path(case_info["image_path"])
    label_path = Path(case_info["label_path"])

    try:
        image_volume = load_nifti_array(image_path).astype(np.float32, copy=False)
        label_volume = load_nifti_array(label_path)
    except Exception as exc:
        print(f"[Warning] Failed to load case {case_id}: {exc}. Skip.")
        return [], {"total": 0, "organ": 0, "background": 0}, []

    label_volume = np.rint(label_volume).astype(np.int64, copy=False)

    if image_volume.ndim != 3 or label_volume.ndim != 3:
        print(
            f"[Warning] Case {case_id} is not 3D: "
            f"image.ndim={image_volume.ndim}, label.ndim={label_volume.ndim}. Skip."
        )
        return [], {"total": 0, "organ": 0, "background": 0}, []

    if image_volume.shape != label_volume.shape:
        print(
            f"[Warning] Shape mismatch for {case_id}: "
            f"image.shape={image_volume.shape}, label.shape={label_volume.shape}. Skip."
        )
        return [], {"total": 0, "organ": 0, "background": 0}, []

    num_slices = image_volume.shape[SLICE_AXIS]
    organ_indices: List[int] = []
    background_indices: List[int] = []
    all_classes: List[int] = []

    for slice_index in range(num_slices):
        mask_slice = extract_slice(label_volume, slice_index, SLICE_AXIS)
        organ_classes = _safe_organ_classes(mask_slice)
        all_classes.extend(organ_classes)
        if organ_classes:
            organ_indices.append(slice_index)
        else:
            background_indices.append(slice_index)

    kept_background = choose_background_indices(background_indices, BACKGROUND_KEEP_RATIO, rng)
    kept_indices = sorted(organ_indices + kept_background)

    slice_rows: List[Dict[str, object]] = []
    preview_count = 0
    for slice_index in kept_indices:
        image_slice = extract_slice(image_volume, slice_index, SLICE_AXIS)
        mask_slice = extract_slice(label_volume, slice_index, SLICE_AXIS)

        image_slice = normalize(image_slice, use_hu_window=USE_HU_WINDOW)
        image_slice = resize_image_slice(image_slice, TARGET_SIZE)
        mask_slice = resize_mask_slice(mask_slice, TARGET_SIZE)

        organ_classes = _safe_organ_classes(mask_slice)
        has_organ = int(len(organ_classes) > 0)
        organ_pixels_total = int((mask_slice > 0).sum())
        slice_name = f"{case_id}_slice{slice_index:04d}.npy"

        image_save_path = output_dirs["images"] / slice_name
        mask_save_path = output_dirs["masks"] / slice_name
        np.save(image_save_path, image_slice.astype(np.float32))
        np.save(mask_save_path, mask_slice.astype(np.int64))

        slice_rows.append(
            {
                "slice_id": slice_name.replace(".npy", ""),
                "case_id": case_id,
                "image_path": str(image_save_path),
                "mask_path": str(mask_save_path),
                "has_organ": has_organ,
                "organ_classes": json.dumps(organ_classes),
                "organ_pixels_total": organ_pixels_total,
            }
        )

        if preview_count < MAX_PREVIEWS_PER_CASE:
            preview_path = output_dirs["previews"] / f"{case_id}_slice{slice_index:04d}.png"
            save_overlay_preview(image_slice, mask_slice, preview_path, case_id, slice_index)
            preview_count += 1

    counts = {
        "total": len(kept_indices),
        "organ": len(organ_indices),
        "background": len(kept_background),
    }
    return slice_rows, counts, sorted(set(all_classes))


def main() -> None:
    print("=== BTCV 2D preprocessing start ===")
    print(f"image_dir: {IMAGE_DIR}")
    print(f"label_dir: {LABEL_DIR}")
    print(f"output_root: {OUTPUT_ROOT}")

    output_dirs = ensure_dirs()
    rng = random.Random(RANDOM_SEED)
    case_mapping = build_case_mapping(IMAGE_DIR, LABEL_DIR)
    print(f"matched case count: {len(case_mapping)}")

    all_slice_rows: List[Dict[str, object]] = []
    detected_classes: List[int] = [0]
    total_slices = 0
    organ_slices = 0
    background_slices = 0

    for idx, case_info in enumerate(case_mapping, start=1):
        case_id = str(case_info["case_id"])
        print(f"[{idx}/{len(case_mapping)}] processing {case_id}")
        slice_rows, counts, classes = process_case(case_info, output_dirs, rng)
        all_slice_rows.extend(slice_rows)
        detected_classes.extend(classes)
        total_slices += counts["total"]
        organ_slices += counts["organ"]
        background_slices += counts["background"]

    write_slice_csv(output_dirs["meta"], all_slice_rows)
    label_map = write_label_map(output_dirs["meta"], detected_classes)
    write_preprocess_config(output_dirs["meta"])

    case_ids = sorted({str(case["case_id"]) for case in case_mapping})
    split_info = split_cases(case_ids)
    with (output_dirs["meta"] / "split.json").open("w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print("=== BTCV 2D preprocessing summary ===")
    print(f"case count: {len(case_ids)}")
    print(f"total slices: {total_slices}")
    print(f"organ slices: {organ_slices}")
    print(f"background slices: {background_slices}")
    print(f"train case count: {len(split_info['train_cases'])}")
    print(f"val case count: {len(split_info['val_cases'])}")
    print(f"test case count: {len(split_info['test_cases'])}")
    print(f"detected classes: {sorted(label_map.keys())}")


if __name__ == "__main__":
    main()
