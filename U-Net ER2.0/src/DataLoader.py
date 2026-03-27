#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates, zoom
from torch.utils.data import Dataset, DataLoader


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


def resize_field_bicubic(field_small: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """
    用 scipy.ndimage.zoom 做三次插值。
    这里对应的是论文里“位移场逐像素用 bicubic interpolation”。
    """
    h, w = out_hw
    zh = h / field_small.shape[0]
    zw = w / field_small.shape[1]
    field = zoom(field_small, zoom=(zh, zw), order=3)

    # zoom 后可能差 1~2 个像素，统一裁到目标大小
    if field.shape[0] < h:
        pad_h = h - field.shape[0]
        field = np.pad(field, ((0, pad_h), (0, 0)), mode="edge")
    if field.shape[1] < w:
        pad_w = w - field.shape[1]
        field = np.pad(field, ((0, 0), (0, pad_w)), mode="edge")

    return field[:h, :w].astype(np.float32)


def elastic_deform_image_and_instance(
    image: np.ndarray,
    instance: np.ndarray,
    displacement_std: float = 10.0,
    grid_size: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对 image 和 instance mask 做同一个 elastic deformation。
    原文核心设定：
    - 3x3 coarse grid
    - displacement ~ N(0, 10px)
    - per-pixel displacement 用 bicubic interpolation
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = image.shape

    dx_small = rng.normal(0.0, displacement_std, size=(grid_size, grid_size)).astype(np.float32)
    dy_small = rng.normal(0.0, displacement_std, size=(grid_size, grid_size)).astype(np.float32)

    dx = resize_field_bicubic(dx_small, (h, w))
    dy = resize_field_bicubic(dy_small, (h, w))

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.array([yy + dy, xx + dx])

    warped_img = map_coordinates(
        image.astype(np.float32, copy=False),
        coords,
        order=1,
        mode="reflect",
    ).reshape(h, w)

    warped_inst = map_coordinates(
        instance.astype(np.float32, copy=False),
        coords,
        order=0,
        mode="nearest",
    ).reshape(h, w)

    warped_inst = np.rint(warped_inst).astype(np.uint16)
    return warped_img.astype(np.float32), warped_inst


def maybe_gray_value_variation(
    image: np.ndarray,
    enabled: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    论文提到 gray value variations，但没有给唯一公式。
    所以这里只做一个很保守的线性灰度扰动。
    """
    if not enabled:
        return image
    if rng is None:
        rng = np.random.default_rng()

    scale = rng.uniform(0.9, 1.1)
    shift = rng.normal(0.0, 0.05 * max(float(image.std()), 1e-6))
    out = image.astype(np.float32) * scale + shift
    return out.astype(np.float32)


def pad_for_valid_conv_sampling(
    image: np.ndarray,
    binary: np.ndarray,
    weight: np.ndarray,
    input_size: int,
    output_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    valid conv 下，输入 patch 比输出 patch 大。
    例如经典 U-Net:
    input  = 572
    output = 388
    margin = 92

    image 需要多 pad 一个 margin，用 reflect mirror。
    binary/weight 只在尺寸不足 output_size 时补 pad。
    """
    margin = (input_size - output_size) // 2
    h, w = binary.shape

    extra_h = max(0, output_size - h)
    extra_w = max(0, output_size - w)

    top = extra_h // 2
    bottom = extra_h - top
    left = extra_w // 2
    right = extra_w - left

    image_pad = np.pad(
        image,
        ((margin + top, margin + bottom), (margin + left, margin + right)),
        mode="reflect",
    )

    # binary pad 出来的部分不是真实标注，给 0
    binary_pad = np.pad(
        binary,
        ((top, bottom), (left, right)),
        mode="constant",
        constant_values=0,
    )

    # weight pad 出来的部分也不该参与学习，给 0
    weight_pad = np.pad(
        weight,
        ((top, bottom), (left, right)),
        mode="constant",
        constant_values=0.0,
    )

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

    image_pad, binary_pad, weight_pad = pad_for_valid_conv_sampling(
        image=image,
        binary=binary,
        weight=weight,
        input_size=input_size,
        output_size=output_size,
    )

    hp, wp = binary_pad.shape
    max_y = hp - output_size
    max_x = wp - output_size

    y0 = int(rng.integers(0, max_y + 1))
    x0 = int(rng.integers(0, max_x + 1))

    image_crop = image_pad[y0 : y0 + input_size, x0 : x0 + input_size]
    binary_crop = binary_pad[y0 : y0 + output_size, x0 : x0 + output_size]
    weight_crop = weight_pad[y0 : y0 + output_size, x0 : x0 + output_size]

    return image_crop, binary_crop, weight_crop


class UNetStrictTrainDataset(Dataset):
    """
    严格围绕 U-Net 原文训练逻辑的数据集：
    1. 读 TIFF 原图
    2. 读 instance mask
    3. 训练时对 full image 做 elastic deformation
    4. 由 instance 动态得到 binary 和 weight map
    5. 再裁 valid-conv 对齐的 572->388 patch
    """

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
        gray_value_aug: bool = False,
        w0: float = 10.0,
        sigma: float = 5.0,
    ):
        super().__init__()

        processed_root = Path(processed_root)
        manifest_dir = processed_root / "manifests"
        train_pairs_path = manifest_dir / "train_pairs.json"
        class_weights_path = manifest_dir / "class_weights.json"

        if not train_pairs_path.exists():
            raise FileNotFoundError(f"Missing train_pairs.json: {train_pairs_path}")
        if not class_weights_path.exists():
            raise FileNotFoundError(f"Missing class_weights.json: {class_weights_path}")

        self.records = load_json(train_pairs_path)
        self.class_weights = load_json(class_weights_path)

        self.processed_root = processed_root
        self.input_size = input_size
        self.output_size = output_size
        self.patches_per_image = patches_per_image
        self.elastic_deform = elastic_deform
        self.displacement_std = displacement_std
        self.grid_size = grid_size
        self.normalize = normalize
        self.gray_value_aug = gray_value_aug
        self.w0 = w0
        self.sigma = sigma

        self.w_fg = float(self.class_weights["foreground"])
        self.w_bg = float(self.class_weights["background"])

        if (self.input_size - self.output_size) % 2 != 0:
            raise ValueError("input_size - output_size must be even")
        if self.input_size <= self.output_size:
            raise ValueError("input_size must be larger than output_size for valid conv")

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_image

    def _resolve_image_path(self, rec: Dict) -> Path:
        # 兼容你前面几个版本的 manifest 字段名
        for key in ["image_copy_tif", "image_tif"]:
            if key in rec:
                return Path(rec[key])
        raise KeyError("No image tif path found in manifest record")

    def _resolve_instance_path(self, rec: Dict) -> Path:
        if "instance_npy" not in rec:
            raise KeyError("No instance_npy found in manifest record")
        return Path(rec["instance_npy"])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rec = self.records[index % len(self.records)]

        image_path = self._resolve_image_path(rec)
        instance_path = self._resolve_instance_path(rec)

        image = read_tif_2d(image_path).astype(np.float32, copy=False)
        instance = np.load(str(instance_path)).astype(np.uint16, copy=False)

        if image.shape != instance.shape:
            raise ValueError(
                f"Shape mismatch: image {image.shape} vs instance {instance.shape} for {image_path.name}"
            )

        rng = np.random.default_rng()

        if self.elastic_deform:
            image, instance = elastic_deform_image_and_instance(
                image=image,
                instance=instance,
                displacement_std=self.displacement_std,
                grid_size=self.grid_size,
                rng=rng,
            )

        image = maybe_gray_value_variation(
            image=image,
            enabled=self.gray_value_aug,
            rng=rng,
        )

        binary = (instance > 0).astype(np.uint8)
        weight = compute_weight_map(
            instance_mask=instance,
            binary_mask=binary,
            w_fg=self.w_fg,
            w_bg=self.w_bg,
            w0=self.w0,
            sigma=self.sigma,
        )

        image_crop, binary_crop, weight_crop = random_valid_pair_crop(
            image=image,
            binary=binary,
            weight=weight,
            input_size=self.input_size,
            output_size=self.output_size,
            rng=rng,
        )

        image_crop = normalize_image(image_crop, self.normalize)

        image_t = torch.from_numpy(image_crop[None, ...].astype(np.float32))
        target_t = torch.from_numpy(binary_crop.astype(np.int64))
        weight_t = torch.from_numpy(weight_crop.astype(np.float32))

        sample = {
            "image": image_t,         # [1, 572, 572]
            "target": target_t,       # [388, 388], long, 0/1
            "weight": weight_t,       # [388, 388], float
            "seq": rec["seq"],
            "frame": rec["frame"],
        }
        return sample


def build_unet_strict_train_loader(
    processed_root: str | Path,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    ds = UNetStrictTrainDataset(
        processed_root=processed_root,
        **dataset_kwargs,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


if __name__ == "__main__":
    root = Path("/Users/dby051225/Desktop/VIS/U-Net/U-Net ER2.0/processed_unet_strict")

    ds = UNetStrictTrainDataset(
        processed_root=root,
        input_size=572,
        output_size=388,
        patches_per_image=2,
        elastic_deform=True,
        displacement_std=10.0,
        grid_size=3,
        normalize="zscore",
        gray_value_aug=False,
        w0=10.0,
        sigma=5.0,
    )

    sample = ds[0]
    print("image :", sample["image"].shape, sample["image"].dtype)
    print("target:", sample["target"].shape, sample["target"].dtype, sample["target"].min().item(), sample["target"].max().item())
    print("weight:", sample["weight"].shape, sample["weight"].dtype, float(sample["weight"].min()), float(sample["weight"].max()))

    loader = build_unet_strict_train_loader(
        processed_root=root,
        batch_size=1,
        num_workers=0,
        input_size=572,
        output_size=388,
        patches_per_image=2,
        elastic_deform=True,
        displacement_std=10.0,
        grid_size=3,
        normalize="zscore",
        gray_value_aug=False,
        w0=10.0,
        sigma=5.0,
    )

    batch = next(iter(loader))
    print("batch image :", batch["image"].shape)
    print("batch target:", batch["target"].shape)
    print("batch weight:", batch["weight"].shape)
