#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_ROOT_DIR = Path(
    "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/Dataset 2D/processed/BTCV_multiorgan_2d"
)


class BTCV2DUNetDataset(Dataset):
    """
    2D U-Net slice dataset for BTCV multi-organ segmentation.

    train:
    - foreground-aware random sampling

    val/test:
    - deterministic sequential access
    """

    def __init__(
        self,
        root_dir: str | Path,
        mode: str,
        input_size: Tuple[int, int] = (256, 256),
        samples_per_epoch: int = 400,
        foreground_ratio: float = 0.7,
        use_augmentation: bool | None = None,
        random_seed: int = 42,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.mode = mode.lower()
        self.input_size = tuple(int(v) for v in input_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.foreground_ratio = float(foreground_ratio)
        self.random_seed = int(random_seed)

        if self.mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.rng = random.Random(self.random_seed)
        self.np_rng = np.random.default_rng(self.random_seed)
        self.meta_dir = self.root_dir / "meta"
        self.slices_csv_path = self.meta_dir / "slices.csv"
        self.split_json_path = self.meta_dir / "split.json"
        self.label_map_path = self.meta_dir / "label_map.json"

        if use_augmentation is None:
            self.use_augmentation = self.mode == "train"
        else:
            self.use_augmentation = bool(use_augmentation)

        self.records = self._load_records()
        self.foreground_records = [row for row in self.records if int(row["has_organ"]) == 1]
        self.background_records = [row for row in self.records if int(row["has_organ"]) == 0]

        if len(self.records) == 0:
            raise RuntimeError(f"No slice records found for mode={self.mode} under {self.root_dir}")

    def _load_records(self) -> List[Dict]:
        if not self.slices_csv_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {self.slices_csv_path}")
        if not self.split_json_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {self.split_json_path}")
        if not self.label_map_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {self.label_map_path}")

        slices_df = pd.read_csv(self.slices_csv_path)
        with self.split_json_path.open("r", encoding="utf-8") as f:
            split_info = json.load(f)

        split_key = f"{self.mode}_cases"
        target_case_ids = set(str(v) for v in split_info.get(split_key, []))
        filtered_df = slices_df[slices_df["case_id"].astype(str).isin(target_case_ids)].copy()
        filtered_df = filtered_df.sort_values(["case_id", "slice_id"]).reset_index(drop=True)

        return filtered_df.to_dict(orient="records")

    def _load_npy(self, row: Dict) -> Tuple[np.ndarray, np.ndarray]:
        image = np.load(Path(row["image_path"])).astype(np.float32, copy=False)
        mask = np.load(Path(row["mask_path"])).astype(np.int64, copy=False)

        if image.ndim != 2 or mask.ndim != 2:
            raise ValueError(
                f"Slice {row['slice_id']} must be 2D, got image.ndim={image.ndim}, mask.ndim={mask.ndim}"
            )
        if image.shape != mask.shape:
            raise ValueError(
                f"Slice shape mismatch for {row['slice_id']}: image.shape={image.shape}, mask.shape={mask.shape}"
            )
        return image, mask

    def _sample_train_row(self) -> Dict:
        use_foreground = self.rng.random() < self.foreground_ratio
        if use_foreground and self.foreground_records:
            return self.foreground_records[self.rng.randrange(len(self.foreground_records))]
        return self.records[self.rng.randrange(len(self.records))]

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.use_augmentation or self.mode != "train":
            return image, mask

        if self.rng.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        if self.rng.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        k = self.rng.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k=k).copy()
            mask = np.rot90(mask, k=k).copy()

        if self.rng.random() < 0.3:
            scale = 1.0 + self.rng.uniform(-0.1, 0.1)
            shift = self.rng.uniform(-0.05, 0.05)
            image = np.clip(image * scale + shift, 0.0, 1.0)

        return image.astype(np.float32, copy=False), mask.astype(np.int64, copy=False)

    def __len__(self) -> int:
        if self.mode == "train":
            return self.samples_per_epoch
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self._sample_train_row() if self.mode == "train" else self.records[index]
        image, mask = self._load_npy(row)
        image, mask = self._augment(image, mask)

        image_tensor = torch.from_numpy(image).unsqueeze(0).to(torch.float32)
        mask_tensor = torch.from_numpy(mask).to(torch.long)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "case_id": str(row["case_id"]),
            "slice_id": str(row["slice_id"]),
        }


def build_btcv2d_unet_dataloader(
    root_dir: str | Path,
    mode: str,
    batch_size: int = 8,
    num_workers: int = 0,
    input_size: Tuple[int, int] = (256, 256),
    samples_per_epoch: int = 400,
    foreground_ratio: float = 0.7,
    use_augmentation: bool | None = None,
    random_seed: int = 42,
    pin_memory: bool = False,
    drop_last: bool | None = None,
) -> DataLoader:
    dataset = BTCV2DUNetDataset(
        root_dir=root_dir,
        mode=mode,
        input_size=input_size,
        samples_per_epoch=samples_per_epoch,
        foreground_ratio=foreground_ratio,
        use_augmentation=use_augmentation,
        random_seed=random_seed,
    )

    mode = mode.lower()
    if drop_last is None:
        drop_last = mode == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


if __name__ == "__main__":
    loader = build_btcv2d_unet_dataloader(
        root_dir=DEFAULT_ROOT_DIR,
        mode="train",
        batch_size=4,
        num_workers=0,
        input_size=(256, 256),
        samples_per_epoch=16,
        foreground_ratio=0.7,
        use_augmentation=True,
        random_seed=42,
    )
    batch = next(iter(loader))
    print("image.shape:", batch["image"].shape)
    print("mask.shape:", batch["mask"].shape)
    print("image.dtype:", batch["image"].dtype)
    print("mask.dtype:", batch["mask"].dtype)
    print("mask.unique()[:20]:", torch.unique(batch["mask"])[:20])
