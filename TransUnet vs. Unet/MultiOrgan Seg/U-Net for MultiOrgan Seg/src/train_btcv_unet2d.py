#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


DATA_ROOT = "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/Dataset 2D/processed/BTCV_multiorgan_2d"
SAVE_ROOT = "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/outputs_2d_unet/BTCV_multiorgan_2d"
NUM_EPOCHS = 30
BATCH_SIZE = 8
NUM_WORKERS = 0
INPUT_SIZE = (320, 320)
MAX_TRAIN_STEPS_PER_EPOCH = 400
MAX_VAL_STEPS_PER_EPOCH = 80
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
BASE_CHANNELS = 64
RANDOM_SEED = 42
IGNORE_BACKGROUND = True
USE_DICE_LOSS = True
CE_WEIGHT = 1.0
DICE_WEIGHT = 1.0


THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from btcv2d_unet_dataloader import build_btcv2d_unet_dataloader


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def load_unet2d_class():
    from unet2d_btcv import UNet2D

    return UNet2D


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_label_map(data_root: str | Path) -> Dict[int, str]:
    label_map_path = Path(data_root) / "meta" / "label_map.json"
    with label_map_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def multiclass_dice_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_background: bool = True,
) -> Tuple[float, float]:
    if preds.ndim == 2:
        preds = preds.unsqueeze(0)
    if targets.ndim == 2:
        targets = targets.unsqueeze(0)

    class_range = range(1, num_classes) if ignore_background else range(num_classes)
    dice_scores: List[float] = []
    iou_scores: List[float] = []

    for batch_idx in range(preds.shape[0]):
        for class_idx in class_range:
            pred_mask = preds[batch_idx] == class_idx
            target_mask = targets[batch_idx] == class_idx

            pred_sum = int(pred_mask.sum().item())
            target_sum = int(target_mask.sum().item())
            if pred_sum == 0 and target_sum == 0:
                continue

            intersection = float((pred_mask & target_mask).sum().item())
            union = float((pred_mask | target_mask).sum().item())
            dice_scores.append((2.0 * intersection) / (pred_sum + target_sum + 1e-8))
            iou_scores.append(intersection / (union + 1e-8))

    if not dice_scores:
        return 0.0, 0.0
    return float(np.mean(dice_scores)), float(np.mean(iou_scores))


def multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_background: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float()

    if ignore_background and num_classes > 1:
        probs = probs[:, 1:, ...]
        one_hot = one_hot[:, 1:, ...]

    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(one_hot, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _compute_total_loss(logits: torch.Tensor, masks: torch.Tensor, ce_criterion: nn.Module) -> torch.Tensor:
    ce_loss = ce_criterion(logits, masks)
    if not USE_DICE_LOSS:
        return CE_WEIGHT * ce_loss
    dice_loss = multiclass_dice_loss(logits, masks, ignore_background=IGNORE_BACKGROUND)
    return CE_WEIGHT * ce_loss + DICE_WEIGHT * dice_loss


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    ce_criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    num_batches = 0

    total_steps = min(len(train_loader), MAX_TRAIN_STEPS_PER_EPOCH)
    progress = tqdm(range(total_steps), total=total_steps, desc=f"train {epoch}", leave=False)
    iterator = iter(train_loader)

    for step_idx in progress:
        batch = next(iterator)
        images = batch["image"].to(device=device, dtype=torch.float32)
        masks = batch["mask"].to(device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = _compute_total_loss(logits, masks, ce_criterion)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        batch_dice, batch_iou = multiclass_dice_iou(
            preds,
            masks,
            num_classes=num_classes,
            ignore_background=IGNORE_BACKGROUND,
        )

        running_loss += float(loss.item())
        running_dice += batch_dice
        running_iou += batch_iou
        num_batches += 1

        progress.set_postfix(
            loss=f"{float(loss.item()):.4f}",
            dice=f"{batch_dice:.4f}",
            iou=f"{batch_iou:.4f}",
        )

    if total_steps < len(train_loader):
        print("train early stop for budget mode")

    return (
        running_loss / max(num_batches, 1),
        running_dice / max(num_batches, 1),
        running_iou / max(num_batches, 1),
    )


@torch.no_grad()
def validate_one_epoch(
    epoch: int,
    model: nn.Module,
    val_loader,
    ce_criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    num_batches = 0

    total_steps = min(len(val_loader), MAX_VAL_STEPS_PER_EPOCH)
    progress = tqdm(range(total_steps), total=total_steps, desc=f"val {epoch}", leave=False)
    iterator = iter(val_loader)

    for step_idx in progress:
        batch = next(iterator)
        images = batch["image"].to(device=device, dtype=torch.float32)
        masks = batch["mask"].to(device=device, dtype=torch.long)

        logits = model(images)
        loss = _compute_total_loss(logits, masks, ce_criterion)
        preds = torch.argmax(logits, dim=1)
        batch_dice, batch_iou = multiclass_dice_iou(
            preds,
            masks,
            num_classes=num_classes,
            ignore_background=IGNORE_BACKGROUND,
        )

        running_loss += float(loss.item())
        running_dice += batch_dice
        running_iou += batch_iou
        num_batches += 1

        progress.set_postfix(
            loss=f"{float(loss.item()):.4f}",
            dice=f"{batch_dice:.4f}",
            iou=f"{batch_iou:.4f}",
        )

    if total_steps < len(val_loader):
        print("val early stop for budget mode")

    return (
        running_loss / max(num_batches, 1),
        running_dice / max(num_batches, 1),
        running_iou / max(num_batches, 1),
    )


def save_checkpoint(
    save_path: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_dice: float,
    best_val_iou: float,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_dice": float(best_val_dice),
            "best_val_iou": float(best_val_iou),
        },
        save_path,
    )


def _case_count(loader) -> int:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return 0
    records = getattr(dataset, "records", [])
    return len({str(row["case_id"]) for row in records})


def _write_train_log(rows: Iterable[Dict[str, object]], save_root: Path) -> None:
    rows = list(rows)
    log_path = save_root / "train_log.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "train_dice",
        "train_iou",
        "val_loss",
        "val_dice",
        "val_iou",
        "is_best",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    set_seed(RANDOM_SEED)

    data_root = Path(DATA_ROOT)
    save_root = Path(SAVE_ROOT)
    save_root.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(data_root)
    num_classes = max(label_map.keys()) + 1

    train_loader = build_btcv2d_unet_dataloader(
        root_dir=data_root,
        mode="train",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        input_size=INPUT_SIZE,
        samples_per_epoch=MAX_TRAIN_STEPS_PER_EPOCH * BATCH_SIZE,
        foreground_ratio=0.7,
        use_augmentation=True,
        random_seed=RANDOM_SEED,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = build_btcv2d_unet_dataloader(
        root_dir=data_root,
        mode="val",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        input_size=INPUT_SIZE,
        samples_per_epoch=MAX_VAL_STEPS_PER_EPOCH * BATCH_SIZE,
        foreground_ratio=0.7,
        use_augmentation=False,
        random_seed=RANDOM_SEED,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )

    UNet2D = load_unet2d_class()
    model = UNet2D(in_channels=1, num_classes=num_classes, base_channels=BASE_CHANNELS).to(DEVICE)
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"device: {DEVICE}")
    print(f"data_root: {data_root}")
    print(f"save_root: {save_root}")
    print(f"num_classes: {num_classes}")
    print(f"train case count: {_case_count(train_loader)}")
    print(f"val case count: {_case_count(val_loader)}")
    print(f"input_size: {INPUT_SIZE}")
    print(f"batch_size: {BATCH_SIZE}")
    print(f"num_epochs: {NUM_EPOCHS}")
    print(f"max train steps: {MAX_TRAIN_STEPS_PER_EPOCH}")
    print(f"max val steps: {MAX_VAL_STEPS_PER_EPOCH}")

    history: List[Dict[str, object]] = []
    best_epoch = 0
    best_val_dice = -1.0
    best_val_iou = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_dice, train_iou = train_one_epoch(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            ce_criterion=ce_criterion,
            device=DEVICE,
            num_classes=num_classes,
        )
        val_loss, val_dice, val_iou = validate_one_epoch(
            epoch=epoch,
            model=model,
            val_loader=val_loader,
            ce_criterion=ce_criterion,
            device=DEVICE,
            num_classes=num_classes,
        )

        is_best = val_dice > best_val_dice
        if is_best:
            best_epoch = epoch
            best_val_dice = val_dice
            best_val_iou = val_iou
            save_checkpoint(save_root / "best.pth", epoch, model, optimizer, best_val_dice, best_val_iou)

        save_checkpoint(save_root / "last.pth", epoch, model, optimizer, best_val_dice, best_val_iou)

        history.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_dice": f"{train_dice:.6f}",
                "train_iou": f"{train_iou:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_dice": f"{val_dice:.6f}",
                "val_iou": f"{val_iou:.6f}",
                "is_best": int(is_best),
            }
        )
        _write_train_log(history, save_root)

        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        print(f"Train | loss={train_loss:.6f} dice={train_dice:.6f} iou={train_iou:.6f}")
        print(f"Val   | loss={val_loss:.6f} dice={val_dice:.6f} iou={val_iou:.6f}")

    print(f"best epoch: {best_epoch}")
    print(f"best val dice: {best_val_dice:.6f}")
    print(f"best val iou: {best_val_iou:.6f}")
    print(f"save root: {save_root}")


if __name__ == "__main__":
    main()
