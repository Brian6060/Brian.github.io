#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates, zoom
from torch.utils.data import DataLoader, Dataset


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_tif_2d(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Only support 2D tif, got {path} with shape {arr.shape}")
    return arr


def normalize_image(img: np.ndarray, mode: Optional[str] = "zscore") -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    if mode is None:
        return img
    if mode == "zscore":
        mean = float(img.mean())
        std = float(img.std())
        if std < 1e-6:
            return img - mean
        return (img - mean) / std
    if mode == "minmax":
        mn = float(img.min())
        mx = float(img.max())
        if mx - mn < 1e-6:
            return img - mn
        return (img - mn) / (mx - mn)
    raise ValueError(f"Unknown normalize mode: {mode}")


def instance_boundaries(instance_mask: np.ndarray) -> List[np.ndarray]:
    labels = [int(x) for x in np.unique(instance_mask) if int(x) > 0]
    boundaries: List[np.ndarray] = []
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
    w0: float = 10.0,
    sigma: float = 5.0,
) -> np.ndarray:
    wc = np.where(binary_mask > 0, w_fg, w_bg).astype(np.float32)
    boundaries = instance_boundaries(instance_mask)
    if len(boundaries) < 2:
        return wc

    dists = [distance_transform_edt(~border) for border in boundaries]
    dist_stack = np.stack(dists, axis=0)
    part = np.partition(dist_stack, kth=1, axis=0)
    d1 = part[0]
    d2 = part[1]
    border_term = w0 * np.exp(-((d1 + d2) ** 2) / (2.0 * (sigma ** 2)))
    return (wc + border_term.astype(np.float32) * (binary_mask == 0).astype(np.float32)).astype(np.float32)


def resize_field_bicubic(field_small: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    zh = h / field_small.shape[0]
    zw = w / field_small.shape[1]
    field = zoom(field_small, zoom=(zh, zw), order=3)
    if field.shape[0] < h:
        field = np.pad(field, ((0, h - field.shape[0]), (0, 0)), mode="edge")
    if field.shape[1] < w:
        field = np.pad(field, ((0, 0), (0, w - field.shape[1])), mode="edge")
    return field[:h, :w].astype(np.float32)


def elastic_deform_image_and_instance(
    image: np.ndarray,
    instance: np.ndarray,
    displacement_std: float = 10.0,
    grid_size: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    h, w = image.shape
    dx_small = rng.normal(0.0, displacement_std, size=(grid_size, grid_size)).astype(np.float32)
    dy_small = rng.normal(0.0, displacement_std, size=(grid_size, grid_size)).astype(np.float32)
    dx = resize_field_bicubic(dx_small, (h, w))
    dy = resize_field_bicubic(dy_small, (h, w))
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.array([yy + dy, xx + dx])

    warped_img = map_coordinates(image.astype(np.float32, copy=False), coords, order=1, mode="reflect").reshape(h, w)
    warped_inst = map_coordinates(instance.astype(np.float32, copy=False), coords, order=0, mode="nearest").reshape(h, w)
    return warped_img.astype(np.float32), np.rint(warped_inst).astype(np.uint16)


def maybe_gray_value_variation(
    image: np.ndarray,
    enabled: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if not enabled:
        return image.astype(np.float32, copy=False)
    if rng is None:
        rng = np.random.default_rng()
    scale = rng.uniform(0.9, 1.1)
    shift = rng.normal(0.0, 0.05 * max(float(image.std()), 1e-6))
    return (image.astype(np.float32) * scale + shift).astype(np.float32)


def pad_for_valid_conv_sampling(
    image: np.ndarray,
    binary: np.ndarray,
    weight: np.ndarray,
    input_size: int,
    output_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    margin = (input_size - output_size) // 2
    h, w = binary.shape
    extra_h = max(0, output_size - h)
    extra_w = max(0, output_size - w)
    top = extra_h // 2
    bottom = extra_h - top
    left = extra_w // 2
    right = extra_w - left

    image_pad = np.pad(image, ((margin + top, margin + bottom), (margin + left, margin + right)), mode="reflect")
    binary_pad = np.pad(binary, ((top, bottom), (left, right)), mode="constant", constant_values=0)
    weight_pad = np.pad(weight, ((top, bottom), (left, right)), mode="constant", constant_values=0.0)
    return image_pad, binary_pad, weight_pad


def random_valid_pair_crop(
    image: np.ndarray,
    binary: np.ndarray,
    weight: np.ndarray,
    input_size: int,
    output_size: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    image_pad, binary_pad, weight_pad = pad_for_valid_conv_sampling(image, binary, weight, input_size, output_size)
    hp, wp = binary_pad.shape
    y0 = int(rng.integers(0, hp - output_size + 1))
    x0 = int(rng.integers(0, wp - output_size + 1))
    image_crop = image_pad[y0:y0 + input_size, x0:x0 + input_size]
    binary_crop = binary_pad[y0:y0 + output_size, x0:x0 + output_size]
    weight_crop = weight_pad[y0:y0 + output_size, x0:x0 + output_size]
    return image_crop, binary_crop, weight_crop


class UNetStrictTrainDataset(Dataset):
    def __init__(
        self,
        processed_root: str | Path,
        input_size: int = 572,
        output_size: int = 388,
        patches_per_image: int = 32,
        elastic_deform: bool = True,
        displacement_std: float = 10.0,
        grid_size: int = 3,
        normalize: Optional[str] = "zscore",
        gray_value_aug: bool = True,
        w0: float = 10.0,
        sigma: float = 5.0,
    ):
        super().__init__()
        self.processed_root = Path(processed_root).expanduser().resolve()
        self.manifest_dir = self.processed_root / "manifests"
        self.manifest_path = self.manifest_dir / "train_pairs.json"
        self.class_weights_path = self.manifest_dir / "class_weights.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {self.manifest_path}")
        if not self.class_weights_path.exists():
            raise FileNotFoundError(f"Missing class_weights.json: {self.class_weights_path}")

        self.records = load_json(self.manifest_path)
        self.class_weights = load_json(self.class_weights_path)
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.patches_per_image = int(patches_per_image)
        self.elastic_deform = bool(elastic_deform)
        self.displacement_std = float(displacement_std)
        self.grid_size = int(grid_size)
        self.normalize = normalize
        self.gray_value_aug = bool(gray_value_aug)
        self.w0 = float(w0)
        self.sigma = float(sigma)
        self.w_fg = float(self.class_weights["foreground"])
        self.w_bg = float(self.class_weights["background"])

        if (self.input_size - self.output_size) % 2 != 0:
            raise ValueError("input_size - output_size must be even")
        if self.input_size <= self.output_size:
            raise ValueError("input_size must be larger than output_size")

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_image

    def _resolve_image_path(self, rec: Dict) -> Path:
        for key in ("image_copy_tif", "image_tif"):
            value = rec.get(key)
            if value:
                path = Path(value)
                if path.exists():
                    return path
        fallback = self.processed_root / "images_tif" / f'{rec["seq"]}_{rec["frame"]}.tif'
        if fallback.exists():
            return fallback
        raise KeyError(f"Manifest record lacks image path: {rec}")

    def _resolve_instance_path(self, rec: Dict) -> Path:
        value = rec.get("instance_npy")
        if not value:
            raise KeyError(f"Manifest record lacks instance_npy: {rec}")
        path = Path(value)
        if path.exists():
            return path
        fallback = self.processed_root / "masks_instance_npy" / f'{rec["seq"]}_{rec["frame"]}.npy'
        if fallback.exists():
            return fallback
        return path

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        rec = self.records[index % len(self.records)]
        image = read_tif_2d(self._resolve_image_path(rec)).astype(np.float32, copy=False)
        instance = np.load(str(self._resolve_instance_path(rec))).astype(np.uint16, copy=False)
        if image.shape != instance.shape:
            raise ValueError(f"Shape mismatch for {rec.get('stem', 'unknown')}: image {image.shape} vs instance {instance.shape}")

        rng = np.random.default_rng()
        if self.elastic_deform:
            image, instance = elastic_deform_image_and_instance(
                image=image,
                instance=instance,
                displacement_std=self.displacement_std,
                grid_size=self.grid_size,
                rng=rng,
            )

        image = maybe_gray_value_variation(image=image, enabled=self.gray_value_aug, rng=rng)
        binary = (instance > 0).astype(np.uint8)
        weight = compute_weight_map(instance, binary, self.w_fg, self.w_bg, self.w0, self.sigma)

        image_crop, binary_crop, weight_crop = random_valid_pair_crop(
            image=image,
            binary=binary,
            weight=weight,
            input_size=self.input_size,
            output_size=self.output_size,
            rng=rng,
        )
        image_crop = normalize_image(image_crop, self.normalize)
        return {
            "image": torch.from_numpy(image_crop[None, ...].astype(np.float32)),
            "target": torch.from_numpy(binary_crop.astype(np.int64)),
            "weight": torch.from_numpy(weight_crop.astype(np.float32)),
            "seq": str(rec["seq"]),
            "frame": str(rec["frame"]),
            "stem": f'{rec["seq"]}_{rec["frame"]}',
        }


def build_unet_strict_train_loader(
    processed_root: str | Path,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    dataset = UNetStrictTrainDataset(processed_root=processed_root, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


if __name__ == "__main__":
    default_root = Path("/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0/processed_unet_train_auto_originalstyle")
    dataset = UNetStrictTrainDataset(
        processed_root=default_root,
        input_size=572,
        output_size=388,
        patches_per_image=2,
        elastic_deform=True,
        gray_value_aug=True,
        w0=10.0,
        sigma=5.0,
    )
    sample = dataset[0]
    print("train manifest:", dataset.manifest_path)
    print("train sample image :", tuple(sample["image"].shape))
    print("train sample target:", tuple(sample["target"].shape))
    print("train sample weight:", tuple(sample["weight"].shape))
