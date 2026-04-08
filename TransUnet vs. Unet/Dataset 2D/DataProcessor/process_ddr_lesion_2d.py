#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


SOURCE_ROOT = Path(
    "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/Dataset 2D/raw/DDR_dataset_unzipped/DDR-dataset"
)
LESION_ROOT = SOURCE_ROOT / "lesion_segmentation"
OUTPUT_ROOT = Path(
    "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/Dataset 2D/processed/DDR_lesion_2d"
)

TARGET_SIZE = (512, 512)
RANDOM_SEED = 42
TASK_TYPE = "2d_multiclass_small_lesion_segmentation"
IMAGE_MODE = "RGB"
IMAGE_NORMALIZATION = "float32_[0,1]"
CLASS_ORDER = ["EX", "HE", "MA", "SE"]
CLASS_TO_LABEL = {"EX": 1, "HE": 2, "MA": 3, "SE": 4}
LABEL_MAP = {
    0: "background",
    1: "EX",
    2: "HE",
    3: "MA",
    4: "SE",
}
MASK_MERGE_PRIORITY = ["MA", "HE", "EX", "SE"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
PREVIEW_COLORS = {
    1: "#f4b400",
    2: "#db4437",
    3: "#00acc1",
    4: "#7cb342",
}
MAX_PREVIEWS_PER_SPLIT = 6


def ensure_dirs() -> Dict[str, Path]:
    images_dir = OUTPUT_ROOT / "images"
    masks_dir = OUTPUT_ROOT / "masks"
    meta_dir = OUTPUT_ROOT / "meta"
    previews_dir = OUTPUT_ROOT / "previews"

    for directory in [OUTPUT_ROOT, images_dir, masks_dir, meta_dir, previews_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Clean previous generated artifacts so reruns stay consistent.
    for path in images_dir.glob("*.npy"):
        path.unlink()
    for path in masks_dir.glob("*.npy"):
        path.unlink()
    for path in previews_dir.glob("*.png"):
        path.unlink()
    for path in meta_dir.glob("*.json"):
        path.unlink()
    for path in meta_dir.glob("*.csv"):
        path.unlink()

    return {
        "images": images_dir,
        "masks": masks_dir,
        "meta": meta_dir,
        "previews": previews_dir,
    }


def normalize_image_id(name_or_path: str | Path) -> str:
    name = Path(name_or_path).name
    suffixes = Path(name).suffixes
    for suffix in suffixes:
        if suffix:
            name = name[: -len(suffix)]

    normalized = name.strip().lower()
    normalized = normalized.replace("_", "-")
    normalized = normalized.replace(" ", "-")

    boundary_tokens = [
        "image",
        "images",
        "img",
        "mask",
        "masks",
        "label",
        "labels",
        "seg",
        "segmentation",
        "lesion",
        "annotation",
        "annot",
    ]

    changed = True
    while changed and normalized:
        changed = False
        for token in boundary_tokens:
            for prefix in (token + "_", token + "-", token):
                if normalized.startswith(prefix) and len(normalized) > len(prefix):
                    normalized = normalized[len(prefix) :]
                    changed = True
            for suffix in ("_" + token, "-" + token, token):
                if normalized.endswith(suffix) and len(normalized) > len(suffix):
                    normalized = normalized[: -len(suffix)]
                    changed = True
        normalized = normalized.strip("_- ")

    normalized = re.sub(r"[^a-z0-9-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized


def list_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        return []
    return sorted(
        [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    )


def detect_split_structure(root: Path) -> Dict[str, Dict[str, object]]:
    print(f"[Info] Scanning lesion_segmentation root: {root}")

    split_aliases = {
        "train": "train",
        "valid": "val",
        "val": "val",
        "test": "test",
    }
    label_dir_candidates = ["label", "segmentation label"]

    detected: Dict[str, Dict[str, object]] = {}
    for split_name, canonical_split in split_aliases.items():
        split_dir = root / split_name
        if not split_dir.exists():
            print(f"[Info] Split '{split_name}' not found.")
            continue

        image_dir = split_dir / "image"
        label_dir = None
        for candidate in label_dir_candidates:
            candidate_dir = split_dir / candidate
            if candidate_dir.exists():
                label_dir = candidate_dir
                break

        class_dirs = {}
        if label_dir is not None:
            for lesion_class in CLASS_ORDER:
                candidate = label_dir / lesion_class
                if candidate.exists():
                    class_dirs[lesion_class] = candidate

        image_count = len(list_images(image_dir))
        label_counts = {
            lesion_class: len(
                [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in MASK_EXTENSIONS]
            )
            for lesion_class, class_dir in class_dirs.items()
        }

        print(f"[Info] Detected split '{split_name}' -> canonical '{canonical_split}'")
        print(f"        image_dir: {image_dir if image_dir.exists() else 'MISSING'}")
        print(f"        label_dir: {label_dir if label_dir is not None else 'MISSING'}")
        print(f"        image_count: {image_count}")
        print(f"        label_counts: {label_counts if label_counts else 'none'}")

        detected[canonical_split] = {
            "source_split": split_name,
            "split_dir": split_dir,
            "image_dir": image_dir,
            "label_dir": label_dir,
            "class_dirs": class_dirs,
            "image_count": image_count,
            "label_counts": label_counts,
        }

    return detected


def collect_mask_mapping(class_dirs: Dict[str, Path]) -> Dict[str, Dict[str, Path]]:
    class_mappings: Dict[str, Dict[str, Path]] = {}
    for lesion_class, class_dir in class_dirs.items():
        mapping: Dict[str, Path] = {}
        collision_count = 0
        for mask_path in sorted(class_dir.iterdir()):
            if not mask_path.is_file() or mask_path.suffix.lower() not in MASK_EXTENSIONS:
                continue
            normalized_id = normalize_image_id(mask_path)
            if normalized_id in mapping:
                collision_count += 1
                continue
            mapping[normalized_id] = mask_path
        if collision_count:
            print(
                f"[Warning] {lesion_class}: detected {collision_count} normalized-id collisions, "
                "keeping first occurrence."
            )
        class_mappings[lesion_class] = mapping
    return class_mappings


def load_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert(IMAGE_MODE)
        return np.asarray(img, dtype=np.uint8)


def load_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as mask:
        mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    return (mask_array > 0).astype(np.uint8)


def merge_masks(mask_dict: Dict[str, np.ndarray]) -> np.ndarray:
    if not mask_dict:
        raise ValueError("mask_dict is empty")

    reference_shape = next(iter(mask_dict.values())).shape
    merged = np.zeros(reference_shape, dtype=np.uint8)

    for lesion_class in reversed(MASK_MERGE_PRIORITY):
        class_mask = (mask_dict[lesion_class] > 0).astype(np.uint8)
        label_value = CLASS_TO_LABEL[lesion_class]
        merged[class_mask > 0] = label_value

    return merged


def resize_image(image_array: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image_array, mode=IMAGE_MODE)
    resized = pil_image.resize(target_size, resample=Image.BILINEAR)
    resized_array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.transpose(resized_array, (2, 0, 1))


def resize_mask(mask_array: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    pil_mask = Image.fromarray(mask_array.astype(np.uint8), mode="L")
    resized = pil_mask.resize(target_size, resample=Image.NEAREST)
    return np.asarray(resized, dtype=np.uint8)


def save_preview(
    image_chw: np.ndarray,
    mask_hw: np.ndarray,
    save_path: Path,
    image_id: str,
    split_name: str,
    present_classes: Sequence[int],
) -> None:
    image_hwc = np.transpose(image_chw, (1, 2, 0))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.clip(image_hwc, 0.0, 1.0))

    for class_idx in sorted(set(int(v) for v in present_classes)):
        binary_mask = (mask_hw == class_idx).astype(np.uint8)
        if binary_mask.sum() == 0:
            continue
        ax.contour(binary_mask, levels=[0.5], colors=[PREVIEW_COLORS[class_idx]], linewidths=1.0)

    present_text = ",".join(str(v) for v in sorted(set(int(v) for v in present_classes))) or "none"
    ax.set_title(f"{image_id} | split={split_name} | classes={present_text}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def split_train_val(train_ids: Sequence[str], val_ratio: float = 0.2, seed: int = RANDOM_SEED) -> Tuple[List[str], List[str]]:
    train_ids = sorted(set(train_ids))
    if not train_ids:
        return [], []

    rng = random.Random(seed)
    shuffled = train_ids[:]
    rng.shuffle(shuffled)

    val_count = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    val_count = min(val_count, len(shuffled) - 1) if len(shuffled) > 1 else 0

    val_ids = sorted(shuffled[:val_count])
    new_train_ids = sorted(shuffled[val_count:])
    return new_train_ids, val_ids


def build_samples(
    detected_splits: Dict[str, Dict[str, object]],
    output_dirs: Dict[str, Path],
) -> Tuple[List[Dict[str, object]], Dict[str, List[str]], List[int]]:
    samples: List[Dict[str, object]] = []
    split_ids: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    detected_labels = set()
    preview_budget = {"train": 0, "val": 0, "test": 0}

    for canonical_split in ["train", "val", "test"]:
        split_info = detected_splits.get(canonical_split)
        if split_info is None:
            continue

        image_dir = Path(split_info["image_dir"])
        class_dirs = dict(split_info["class_dirs"])
        source_split = str(split_info["source_split"])
        mask_mapping = collect_mask_mapping(class_dirs)
        images = list_images(image_dir)

        print(f"[Info] Processing split='{canonical_split}' from source='{source_split}' with {len(images)} images.")

        for image_path in images:
            try:
                raw_image = load_image(image_path)
                original_height, original_width = raw_image.shape[:2]
                image_id = normalize_image_id(image_path)

                class_masks: Dict[str, np.ndarray] = {}
                any_mask_found = False
                for lesion_class in CLASS_ORDER:
                    mask_path = mask_mapping.get(lesion_class, {}).get(image_id)
                    if mask_path is None:
                        class_masks[lesion_class] = np.zeros((original_height, original_width), dtype=np.uint8)
                        continue

                    loaded_mask = load_mask(mask_path)
                    if loaded_mask.shape != (original_height, original_width):
                        pil_mask = Image.fromarray(loaded_mask.astype(np.uint8), mode="L")
                        pil_mask = pil_mask.resize((original_width, original_height), resample=Image.NEAREST)
                        loaded_mask = np.asarray(pil_mask, dtype=np.uint8)
                    class_masks[lesion_class] = loaded_mask
                    if loaded_mask.sum() > 0:
                        any_mask_found = True

                if not any_mask_found:
                    print(f"[Warning] All four lesion masks missing or empty for '{image_path.name}', skipping sample.")
                    continue

                merged_mask = merge_masks(class_masks)
                processed_image = resize_image(raw_image, TARGET_SIZE).astype(np.float32)
                processed_mask = resize_mask(merged_mask, TARGET_SIZE).astype(np.uint8)

                present_classes = [int(v) for v in np.unique(processed_mask) if int(v) > 0]
                has_lesion = int(len(present_classes) > 0)
                lesion_pixels_total = int((processed_mask > 0).sum())
                ex_pixels = int((processed_mask == 1).sum())
                he_pixels = int((processed_mask == 2).sum())
                ma_pixels = int((processed_mask == 3).sum())
                se_pixels = int((processed_mask == 4).sum())

                image_save_path = output_dirs["images"] / f"{image_id}.npy"
                mask_save_path = output_dirs["masks"] / f"{image_id}.npy"
                np.save(image_save_path, processed_image)
                np.save(mask_save_path, processed_mask)

                samples.append(
                    {
                        "image_id": image_id,
                        "split_source": source_split,
                        "image_path": str(image_save_path),
                        "mask_path": str(mask_save_path),
                        "has_lesion": has_lesion,
                        "present_classes": ",".join(str(v) for v in present_classes),
                        "lesion_pixels_total": lesion_pixels_total,
                        "ex_pixels": ex_pixels,
                        "he_pixels": he_pixels,
                        "ma_pixels": ma_pixels,
                        "se_pixels": se_pixels,
                        "image_width": original_width,
                        "image_height": original_height,
                    }
                )

                split_ids[canonical_split].append(image_id)
                detected_labels.update(present_classes)

                if preview_budget[canonical_split] < MAX_PREVIEWS_PER_SPLIT:
                    preview_path = output_dirs["previews"] / f"{canonical_split}_{image_id}.png"
                    save_preview(
                        image_chw=processed_image,
                        mask_hw=processed_mask,
                        save_path=preview_path,
                        image_id=image_id,
                        split_name=canonical_split,
                        present_classes=present_classes,
                    )
                    preview_budget[canonical_split] += 1

            except Exception as exc:
                print(f"[Warning] Failed to process sample '{image_path}': {exc}")
                continue

    return samples, split_ids, sorted(detected_labels)


def write_label_map(meta_dir: Path) -> None:
    with (meta_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in LABEL_MAP.items()}, f, indent=2, ensure_ascii=False)


def write_preprocess_config(meta_dir: Path, detected_splits: Dict[str, Dict[str, object]]) -> None:
    config = {
        "source_root": str(SOURCE_ROOT),
        "lesion_segmentation_root": str(LESION_ROOT),
        "target_size": list(TARGET_SIZE),
        "image_normalization": IMAGE_NORMALIZATION,
        "image_mode": IMAGE_MODE,
        "mask_merge_priority": MASK_MERGE_PRIORITY,
        "random_seed": RANDOM_SEED,
        "task_type": TASK_TYPE,
        "num_classes": len(LABEL_MAP),
        "detected_splits": {
            split_name: {
                "source_split": split_info["source_split"],
                "image_dir": str(split_info["image_dir"]),
                "label_dir": str(split_info["label_dir"]) if split_info["label_dir"] is not None else None,
                "label_subdirs": {
                    lesion_class: str(path) for lesion_class, path in dict(split_info["class_dirs"]).items()
                },
                "image_count": split_info["image_count"],
                "label_counts": split_info["label_counts"],
            }
            for split_name, split_info in detected_splits.items()
        },
    }

    with (meta_dir / "preprocess_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def write_samples_csv(meta_dir: Path, samples: Sequence[Dict[str, object]]) -> None:
    df = pd.DataFrame(samples)
    if not df.empty:
        df = df.sort_values(by=["split_source", "image_id"]).reset_index(drop=True)
    df.to_csv(meta_dir / "samples.csv", index=False)


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    output_dirs = ensure_dirs()
    detected_splits = detect_split_structure(LESION_ROOT)

    if not detected_splits:
        print("[Error] No valid split structure detected. Exiting.")
        return

    samples, split_ids, detected_labels = build_samples(detected_splits, output_dirs)

    if not split_ids["val"] and split_ids["train"] and "val" not in detected_splits:
        print("[Info] Official val split not found. Creating val split from train with 8:2 ratio.")
        new_train_ids, val_ids = split_train_val(split_ids["train"], val_ratio=0.2, seed=RANDOM_SEED)
        split_ids["train"] = new_train_ids
        split_ids["val"] = val_ids

    split_payload = {
        "train_ids": sorted(split_ids["train"]),
        "val_ids": sorted(split_ids["val"]),
        "test_ids": sorted(split_ids["test"]),
    }

    with (output_dirs["meta"] / "split.json").open("w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2, ensure_ascii=False)

    write_label_map(output_dirs["meta"])
    write_preprocess_config(output_dirs["meta"], detected_splits)
    write_samples_csv(output_dirs["meta"], samples)

    lesion_positive_count = sum(int(sample["has_lesion"]) for sample in samples)
    lesion_negative_count = len(samples) - lesion_positive_count

    print("\n[Summary]")
    print(f"total images processed: {len(samples)}")
    print(f"train count: {len(split_payload['train_ids'])}")
    print(f"val count: {len(split_payload['val_ids'])}")
    print(f"test count: {len(split_payload['test_ids'])}")
    print(f"lesion-positive count: {lesion_positive_count}")
    print(f"lesion-negative count: {lesion_negative_count}")
    print(f"label classes detected: {detected_labels}")
    print(f"detected split structure: {list(detected_splits.keys())}")


if __name__ == "__main__":
    main()
