#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tifffile
from scipy.ndimage import binary_erosion, distance_transform_edt


SEG_NAME_RE = re.compile(r"man_seg(\d+)\.tif$", re.IGNORECASE)
IMG_NAME_RE = re.compile(r"t(\d+)\.tif$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified preprocessing for U-Net train/infer.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "infer"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--w0", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=5.0)
    return parser.parse_args()


def ensure_dirs(out_root: Path, mode: str) -> Dict[str, Path]:
    paths = {
        "images_tif": out_root / "images_tif",
        "manifest": out_root / "manifests",
    }

    if mode == "train":
        paths.update({
            "instance_npy": out_root / "masks_instance_npy",
            "binary_npy": out_root / "masks_binary_npy",
            "weight_npy": out_root / "weight_maps_npy",
        })

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def read_2d_tif(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Only support 2D tif. Got {path} with shape {arr.shape}")
    return arr


def find_train_pairs(data_root: Path) -> List[Tuple[str, str, Path, Path]]:
    pairs = []

    for seq in ["01", "02"]:
        img_dir = data_root / seq
        seg_dir = data_root / f"{seq}_GT" / "SEG"

        if not img_dir.exists():
            raise FileNotFoundError(f"Missing image dir: {img_dir}")
        if not seg_dir.exists():
            raise FileNotFoundError(f"Missing GT SEG dir: {seg_dir}")

        seg_files = sorted(seg_dir.glob("man_seg*.tif"))
        if not seg_files:
            raise FileNotFoundError(f"No man_seg*.tif found in {seg_dir}")

        for seg_path in seg_files:
            m = SEG_NAME_RE.search(seg_path.name)
            if not m:
                continue
            frame = m.group(1)

            img_path = img_dir / f"t{frame}.tif"
            if not img_path.exists():
                raise FileNotFoundError(f"Missing matching image: {img_path}")

            pairs.append((seq, frame, img_path, seg_path))

    if not pairs:
        raise RuntimeError("No train pairs found.")
    return pairs


def find_infer_images(data_root: Path) -> List[Tuple[str, str, Path]]:
    items = []

    for seq in ["01", "02"]:
        img_dir = data_root / seq
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing image dir: {img_dir}")

        for img_path in sorted(img_dir.glob("t*.tif")):
            m = IMG_NAME_RE.search(img_path.name)
            if not m:
                continue
            frame = m.group(1)
            items.append((seq, frame, img_path))

    if not items:
        raise RuntimeError("No test images found.")
    return items


def compute_class_balance(pairs: List[Tuple[str, str, Path, Path]]) -> Dict[str, float]:
    total_fg = 0
    total_bg = 0

    for _, _, _, seg_path in pairs:
        seg = read_2d_tif(seg_path)
        binary = seg > 0
        total_fg += int(binary.sum())
        total_bg += int(binary.size - binary.sum())

    total = total_fg + total_bg
    if total_fg == 0 or total_bg == 0:
        raise ValueError("Foreground or background total pixels is 0.")

    w_fg = total / (2.0 * total_fg)
    w_bg = total / (2.0 * total_bg)

    return {
        "foreground": float(w_fg),
        "background": float(w_bg),
        "total_fg_pixels": int(total_fg),
        "total_bg_pixels": int(total_bg),
        "total_pixels": int(total),
    }


def instance_boundaries(instance_mask: np.ndarray) -> List[np.ndarray]:
    labels = [int(x) for x in np.unique(instance_mask) if int(x) > 0]
    boundaries = []
    structure = np.ones((3, 3), dtype=bool)

    for lab in labels:
        cell = instance_mask == lab
        if not np.any(cell):
            continue
        eroded = binary_erosion(cell, structure=structure, border_value=0)
        border = np.logical_xor(cell, eroded)
        if not np.any(border):
            border = cell
        boundaries.append(border)

    return boundaries


def compute_weight_map(
    instance_mask: np.ndarray,
    binary_mask: np.ndarray,
    w_fg: float,
    w_bg: float,
    w0: float,
    sigma: float,
) -> np.ndarray:
    wc = np.where(binary_mask > 0, w_fg, w_bg).astype(np.float32)

    boundaries = instance_boundaries(instance_mask)
    if len(boundaries) < 2:
        return wc

    dists = []
    for border in boundaries:
        dist = distance_transform_edt(~border)
        dists.append(dist)

    dist_stack = np.stack(dists, axis=0)
    part = np.partition(dist_stack, kth=1, axis=0)
    d1 = part[0]
    d2 = part[1]

    border_term = w0 * np.exp(-((d1 + d2) ** 2) / (2.0 * (sigma ** 2)))
    border_term = border_term.astype(np.float32)

    weights = wc + border_term * (binary_mask == 0).astype(np.float32)
    return weights.astype(np.float32)


def save_npy(path: Path, array: np.ndarray) -> None:
    np.save(str(path), array)


def process_train(data_root: Path, out_root: Path, w0: float, sigma: float) -> None:
    out_dirs = ensure_dirs(out_root, mode="train")
    pairs = find_train_pairs(data_root)
    class_balance = compute_class_balance(pairs)

    w_fg = float(class_balance["foreground"])
    w_bg = float(class_balance["background"])

    records = []

    for seq, frame, img_path, seg_path in pairs:
        img = read_2d_tif(img_path)
        seg = read_2d_tif(seg_path)

        if img.shape != seg.shape:
            raise ValueError(f"Shape mismatch: {img.shape} vs {seg.shape}")

        instance_arr = seg.astype(np.uint16, copy=False)
        binary_arr = (instance_arr > 0).astype(np.uint8)
        weight_arr = compute_weight_map(
            instance_mask=instance_arr,
            binary_mask=binary_arr,
            w_fg=w_fg,
            w_bg=w_bg,
            w0=w0,
            sigma=sigma,
        )

        stem = f"{seq}_{frame}"
        image_copy_tif = out_dirs["images_tif"] / f"{stem}.tif"
        instance_npy = out_dirs["instance_npy"] / f"{stem}.npy"
        binary_npy = out_dirs["binary_npy"] / f"{stem}.npy"
        weight_npy = out_dirs["weight_npy"] / f"{stem}.npy"

        shutil.copy2(img_path, image_copy_tif)
        save_npy(instance_npy, instance_arr)
        save_npy(binary_npy, binary_arr)
        save_npy(weight_npy, weight_arr)

        records.append({
            "seq": seq,
            "frame": frame,
            "image_tif": str(img_path),
            "seg_tif": str(seg_path),
            "image_copy_tif": str(image_copy_tif),
            "instance_npy": str(instance_npy),
            "binary_npy": str(binary_npy),
            "weight_npy": str(weight_npy),
            "shape_hw": [int(img.shape[0]), int(img.shape[1])],
            "image_dtype": str(img.dtype),
            "seg_dtype": str(seg.dtype),
        })

    (out_dirs["manifest"] / "train_pairs.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (out_dirs["manifest"] / "class_weights.json").write_text(
        json.dumps(class_balance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "mode": "train",
        "data_root": str(data_root),
        "out_root": str(out_root),
        "num_pairs": len(records),
        "w0": w0,
        "sigma": sigma,
        "class_balance": class_balance,
    }

    (out_dirs["manifest"] / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Train preprocessing done.")
    print(f"num_pairs = {len(records)}")
    print(f"out_root  = {out_root}")


def process_infer(data_root: Path, out_root: Path) -> None:
    out_dirs = ensure_dirs(out_root, mode="infer")
    items = find_infer_images(data_root)

    records = []

    for seq, frame, img_path in items:
        img = read_2d_tif(img_path)
        stem = f"{seq}_{frame}"
        image_copy_tif = out_dirs["images_tif"] / f"{stem}.tif"

        shutil.copy2(img_path, image_copy_tif)

        records.append({
            "seq": seq,
            "frame": frame,
            "image_tif": str(img_path),
            "image_copy_tif": str(image_copy_tif),
            "shape_hw": [int(img.shape[0]), int(img.shape[1])],
            "image_dtype": str(img.dtype),
        })

    (out_dirs["manifest"] / "test_images.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "mode": "infer",
        "data_root": str(data_root),
        "out_root": str(out_root),
        "num_images": len(records),
        "note": "Inference-only preprocessing. No GT, no masks, no weight maps.",
    }

    (out_dirs["manifest"] / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Infer preprocessing done.")
    print(f"num_images = {len(records)}")
    print(f"out_root   = {out_root}")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()

    if args.mode == "train":
        process_train(data_root, out_root, args.w0, args.sigma)
    elif args.mode == "infer":
        process_infer(data_root, out_root)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
