from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALBUMENTATIONS = True
except Exception:
    A = None
    ToTensorV2 = None
    _HAS_ALBUMENTATIONS = False


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DatasetPaths:
    processed_root: Path
    images_dir: Path
    masks_dir: Path
    splits_dir: Path
    meta_dir: Path
    bbox_index_path: Path


def build_dataset_paths(processed_root: str | Path) -> DatasetPaths:
    root = Path(processed_root).expanduser().resolve()
    return DatasetPaths(
        processed_root=root,
        images_dir=root / "images",
        masks_dir=root / "masks",
        splits_dir=root / "splits",
        meta_dir=root / "meta",
        bbox_index_path=root / "meta" / "bboxes_index.json",
    )


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_split_csv(splits_dir: Path, split: str) -> Path:
    split = split.lower().strip()
    path = splits_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split CSV not found: {path}")
    return path


def _find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def parse_split_dataframe(df: pd.DataFrame, images_dir: Path, masks_dir: Path) -> List[Dict[str, Any]]:
    stem_col = _find_first_existing_column(df, ["stem", "id", "sample_id", "name"])
    image_col = _find_first_existing_column(df, ["image_path", "image", "image_name", "image_file", "img_path"])
    mask_col = _find_first_existing_column(df, ["mask_path", "mask", "mask_name", "mask_file"])

    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        stem: Optional[str] = None
        image_path: Optional[Path] = None
        mask_path: Optional[Path] = None

        if stem_col is not None and pd.notna(row[stem_col]):
            stem = str(row[stem_col]).strip()

        if image_col is not None and pd.notna(row[image_col]):
            image_value = str(row[image_col]).strip()
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = images_dir / image_value
            if stem is None:
                stem = image_path.stem

        if mask_col is not None and pd.notna(row[mask_col]):
            mask_value = str(row[mask_col]).strip()
            mask_path = Path(mask_value)
            if not mask_path.is_absolute():
                mask_path = masks_dir / mask_value

        if stem is None:
            raise ValueError("Unable to infer sample stem from split CSV.")

        if image_path is None:
            candidates = list(images_dir.glob(f"{stem}.*"))
            if len(candidates) != 1:
                raise FileNotFoundError(f"Cannot uniquely resolve image for stem={stem}")
            image_path = candidates[0]

        if mask_path is None:
            candidates = list(masks_dir.glob(f"{stem}.*"))
            if len(candidates) != 1:
                raise FileNotFoundError(f"Cannot uniquely resolve mask for stem={stem}")
            mask_path = candidates[0]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        records.append(
            {
                "stem": stem,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            }
        )

    return records


class KvasirSegDataset(Dataset):
    def __init__(
        self,
        processed_root: str | Path,
        split: str,
        image_size: int = 224,
        train: bool = False,
        use_bbox: bool = True,
    ) -> None:
        super().__init__()
        self.paths = build_dataset_paths(processed_root)
        self.split = split.lower().strip()
        self.image_size = int(image_size)
        self.train = bool(train)
        self.use_bbox = bool(use_bbox)

        split_csv = resolve_split_csv(self.paths.splits_dir, self.split)
        df = pd.read_csv(split_csv)
        self.samples = parse_split_dataframe(df, self.paths.images_dir, self.paths.masks_dir)

        self.bbox_index: Dict[str, Any] = {}
        if self.use_bbox:
            self.bbox_index = load_json(self.paths.bbox_index_path)

        self.albu_transform = self._build_albu_transform(train=self.train, image_size=self.image_size)

        self.image_resize = transforms.Resize((self.image_size, self.image_size), interpolation=Image.BILINEAR)
        self.mask_resize = transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def _build_albu_transform(self, train: bool, image_size: int):
        if not _HAS_ALBUMENTATIONS:
            return None

        if train:
            return A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.15),
                    A.Rotate(limit=12, border_mode=0, p=0.25),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.12,
                        contrast_limit=0.12,
                        p=0.25,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=6,
                        sat_shift_limit=8,
                        val_shift_limit=6,
                        p=0.15,
                    ),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]
            )

        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_rgb(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    @staticmethod
    def _load_mask(path: Path) -> Image.Image:
        return Image.open(path).convert("L")

    @staticmethod
    def _binarize_mask(mask_arr: np.ndarray) -> np.ndarray:
        return (mask_arr > 0).astype(np.float32)

    def _apply_torchvision_transform(self, image: Image.Image, mask: Image.Image) -> Tuple[Tensor, Tensor]:
        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() < 0.15:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            if random.random() < 0.25:
                angle = random.uniform(-12.0, 12.0)
                image = TF.rotate(image, angle=angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
                mask = TF.rotate(mask, angle=angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)

        image = self.image_resize(image)
        mask = self.mask_resize(mask)

        image_t = self.to_tensor(image)
        image_t = self.normalize(image_t)

        mask_t = self.to_tensor(mask)
        mask_t = (mask_t > 0.5).float()
        return image_t, mask_t

    def _apply_transform(self, image: Image.Image, mask: Image.Image) -> Tuple[Tensor, Tensor]:
        if self.albu_transform is not None:
            image_np = np.asarray(image)
            mask_np = np.asarray(mask)
            mask_np = self._binarize_mask(mask_np)

            out = self.albu_transform(image=image_np, mask=mask_np)
            image_t: Tensor = out["image"].float()
            mask_t: Tensor = out["mask"].float()

            if mask_t.ndim == 2:
                mask_t = mask_t.unsqueeze(0)
            elif mask_t.ndim == 3 and mask_t.shape[0] != 1:
                mask_t = mask_t[:1]

            mask_t = (mask_t > 0.5).float()
            return image_t, mask_t

        return self._apply_torchvision_transform(image, mask)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        stem = sample["stem"]
        image_path = Path(sample["image_path"])
        mask_path = Path(sample["mask_path"])

        image = self._load_rgb(image_path)
        mask = self._load_mask(mask_path)

        original_width, original_height = image.size
        image_t, mask_t = self._apply_transform(image, mask)

        meta: Dict[str, Any] = {
            "stem": stem,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "original_size": (original_height, original_width),
            "split": self.split,
        }

        if self.use_bbox and stem in self.bbox_index:
            meta["bbox_info"] = self.bbox_index[stem]

        return {
            "image": image_t,
            "mask": mask_t,
            "meta": meta,
        }


def default_collate_with_meta(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([x["image"] for x in batch], dim=0)
    masks = torch.stack([x["mask"] for x in batch], dim=0)
    metas = [x["meta"] for x in batch]
    return {
        "image": images,
        "mask": masks,
        "meta": metas,
    }


def build_vitunet_dataloader(
    processed_root: str | Path,
    split: str,
    image_size: int = 224,
    batch_size: int = 8,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    use_bbox: bool = True,
) -> DataLoader:
    split = split.lower().strip()
    is_train = split == "train"

    dataset = KvasirSegDataset(
        processed_root=processed_root,
        split=split,
        image_size=image_size,
        train=is_train,
        use_bbox=use_bbox,
    )

    if shuffle is None:
        shuffle = is_train

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=default_collate_with_meta,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )


def _debug_save_sample(batch: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    image = batch["image"][0].detach().cpu()
    mask = batch["mask"][0].detach().cpu()
    meta = batch["meta"][0]

    img = image.clone()
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        img[c] = img[c] * s + m
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    mask_np = mask.squeeze(0).numpy()

    Image.fromarray((img * 255).astype(np.uint8)).save(out_dir / f"{meta['stem']}_image.png")
    Image.fromarray((mask_np * 255).astype(np.uint8)).save(out_dir / f"{meta['stem']}_mask.png")


if __name__ == "__main__":
    processed_root = Path(
        "/Users/brian/Desktop/VCL318/Swin-ViT/ViTUnet vs SwinUnet/Dataset/processed/Kvasir-SEG"
    )

    train_loader = build_vitunet_dataloader(
        processed_root=processed_root,
        split="train",
        image_size=224,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        use_bbox=True,
    )

    batch = next(iter(train_loader))
    print("image shape:", batch["image"].shape, batch["image"].dtype)
    print("mask shape :", batch["mask"].shape, batch["mask"].dtype)
    print("image range:", float(batch["image"].min()), float(batch["image"].max()))
    print("mask unique:", torch.unique(batch["mask"]))
    print("meta keys  :", batch["meta"][0].keys())

    debug_dir = processed_root.parent.parent / "debug" / "vitunet_dataloader"
    _debug_save_sample(batch, debug_dir)
    print(f"Debug sample saved to: {debug_dir}")