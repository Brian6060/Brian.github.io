from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataloader import build_swinunet_dataloader
from model import SwinUNet


DEFAULT_PROCESSED_ROOT = Path(
    "/Users/brian/Desktop/VCL318/Swin-ViT/ViTUnet vs SwinUnet/Dataset/processed/Kvasir-SEG"
)
DEFAULT_SAVE_ROOT = Path(
    "/Users/brian/Desktop/VCL318/Swin-ViT/ViTUnet vs SwinUnet/SwinUnet/runs"
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available.")
    if device_arg == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available.")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def binary_iou_from_probs(prob: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    pred = (prob >= threshold).astype(np.uint8)
    gt = (target >= 0.5).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(_safe_div(float(inter), float(union)))


def binary_dice_from_probs(prob: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    pred = (prob >= threshold).astype(np.uint8)
    gt = (target >= 0.5).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float(_safe_div(float(2 * inter), float(denom)))


def binary_ap_from_probs(prob: np.ndarray, target: np.ndarray) -> float:
    y_true = target.astype(np.uint8).reshape(-1)
    y_score = prob.astype(np.float32).reshape(-1)
    if np.unique(y_true).size == 1:
        if y_true[0] == 0:
            return 1.0 if float(y_score.max()) < 0.5 else 0.0
        return 1.0
    return float(average_precision_score(y_true, y_score))


@torch.no_grad()
def compute_batch_metrics(logits: Tensor, targets: Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    gts = targets.detach().cpu().numpy()

    ious, dices, aps = [], [], []
    for i in range(probs.shape[0]):
        prob = probs[i, 0]
        gt = gts[i, 0]
        ious.append(binary_iou_from_probs(prob, gt, threshold))
        dices.append(binary_dice_from_probs(prob, gt, threshold))
        aps.append(binary_ap_from_probs(prob, gt))

    return {
        "iou": float(np.mean(ious)),
        "dice": float(np.mean(dices)),
        "ap": float(np.mean(aps)),
    }


def build_run_dirs(save_root: Path, run_name: str) -> Dict[str, Path]:
    run_dir = save_root / run_name
    dirs = {
        "run_dir": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "logs": run_dir / "logs",
        "metrics": run_dir / "metrics",
        "overlays": run_dir / "overlays",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def save_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_csv_row(csv_path: Path, header: List[str], row: Dict[str, Any]) -> None:
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_dice: float,
    config: Dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_dice": best_dice,
            "config": config,
        },
        path,
    )


def denormalize_image(image_tensor: Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().clone()
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        image[c] = image[c] * s + m
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return (image * 255).astype(np.uint8)


def make_overlay(image_rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    overlay = image_rgb.copy()

    gt_region = gt_mask.astype(bool)
    overlay[gt_region] = (
        (1 - alpha) * overlay[gt_region] + alpha * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)

    pred_region = pred_mask.astype(bool)
    overlay[pred_region] = (
        (1 - alpha) * overlay[pred_region] + alpha * np.array([255, 0, 0], dtype=np.float32)
    ).astype(np.uint8)

    return overlay


@torch.no_grad()
def save_val_overlays(
    model: nn.Module,
    loader,
    device: torch.device,
    out_dir: Path,
    max_samples: int = 6,
    threshold: float = 0.5,
) -> None:
    model.eval()
    saved = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        metas = batch["meta"]

        logits = model(images)
        probs = torch.sigmoid(logits)

        for i in range(images.size(0)):
            image_rgb = denormalize_image(images[i])
            gt_mask = masks[i, 0].detach().cpu().numpy() >= 0.5
            pred_mask = probs[i, 0].detach().cpu().numpy() >= threshold

            overlay = make_overlay(image_rgb, gt_mask, pred_mask)
            stem = metas[i]["stem"]

            Image.fromarray(image_rgb).save(out_dir / f"{stem}_image.png")
            Image.fromarray((gt_mask.astype(np.uint8) * 255)).save(out_dir / f"{stem}_gt.png")
            Image.fromarray((pred_mask.astype(np.uint8) * 255)).save(out_dir / f"{stem}_pred.png")
            Image.fromarray(overlay).save(out_dir / f"{stem}_overlay.png")

            saved += 1
            if saved >= max_samples:
                return


def train_one_epoch(model, loader, optimizer, criterion, device, max_steps=None) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    for step, batch in enumerate(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        total += bs

        if max_steps is not None and step + 1 >= max_steps:
            break

    return running_loss / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float = 0.5, max_steps=None) -> Dict[str, float]:
    model.eval()

    running_loss = 0.0
    total = 0
    ious, dices, aps = [], [], []

    for step, batch in enumerate(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        bs = images.size(0)
        running_loss += loss.item() * bs
        total += bs

        m = compute_batch_metrics(logits, masks, threshold)
        ious.append(m["iou"])
        dices.append(m["dice"])
        aps.append(m["ap"])

        if max_steps is not None and step + 1 >= max_steps:
            break

    return {
        "loss": running_loss / max(total, 1),
        "iou": float(np.mean(ious)) if ious else 0.0,
        "dice": float(np.mean(dices)) if dices else 0.0,
        "ap": float(np.mean(aps)) if aps else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SwinUNet on Kvasir-SEG")
    parser.add_argument("--processed_root", type=str, default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--save_root", type=str, default=str(DEFAULT_SAVE_ROOT))
    parser.add_argument("--run_name", type=str, default="swinunet_kvasir_seg")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--backbone", type=str, default="swin_tiny_patch4_window7_224")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("1. parsed args", flush=True)
    set_seed(args.seed)
    print("2. seed set", flush=True)

    device = get_device(args.device)
    print(f"3. device = {device}", flush=True)

    processed_root = Path(args.processed_root).expanduser().resolve()
    save_root = Path(args.save_root).expanduser().resolve()
    run_dirs = build_run_dirs(save_root, args.run_name)
    print("4. run dirs ready", flush=True)

    experiment_config = {
        "task": "binary_segmentation",
        "dataset": "Kvasir-SEG",
        "processed_root": str(processed_root),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": str(device),
        "threshold": args.threshold,
        "patience": args.patience,
        "model_name": "SwinUNet",
        "backbone": args.backbone,
        "pretrained": args.pretrained,
        "loss": "0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "smoke_test": args.smoke_test,
    }
    save_json(experiment_config, run_dirs["metrics"] / "experiment_config.json")
    print("5. experiment config saved", flush=True)

    print("6. building dataloaders", flush=True)
    train_loader = build_swinunet_dataloader(
        processed_root=processed_root,
        split="train",
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        use_bbox=True,
    )
    val_loader = build_swinunet_dataloader(
        processed_root=processed_root,
        split="val",
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        use_bbox=True,
    )
    print("7. dataloaders ready", flush=True)

    print("8. building model", flush=True)
    model = SwinUNet(
        backbone=args.backbone,
        pretrained=args.pretrained,
        num_classes=1,
    )
    print("9. model object created", flush=True)

    model = model.to(device)
    print("10. model moved to device", flush=True)

    criterion = BCEDiceLoss(0.5, 0.5)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    print("11. optimizer ready", flush=True)

    log_csv = run_dirs["logs"] / "train_log.csv"
    header = [
        "epoch",
        "lr",
        "train_loss",
        "val_loss",
        "val_iou",
        "val_dice",
        "val_ap",
        "epoch_time_sec",
    ]

    best_dice = -1.0
    best_epoch = -1
    no_improve = 0
    start_all = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        if args.smoke_test and epoch > 1:
            break

        max_steps = 2 if args.smoke_test else None
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, max_steps=max_steps)
        val_metrics = evaluate(model, val_loader, criterion, device, args.threshold, max_steps=max_steps)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": f"{lr:.8f}",
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_metrics['loss']:.6f}",
            "val_iou": f"{val_metrics['iou']:.6f}",
            "val_dice": f"{val_metrics['dice']:.6f}",
            "val_ap": f"{val_metrics['ap']:.6f}",
            "epoch_time_sec": f"{epoch_time:.3f}",
        }
        append_csv_row(log_csv, header, row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_iou={val_metrics['iou']:.6f} | "
            f"val_dice={val_metrics['dice']:.6f} | "
            f"val_ap={val_metrics['ap']:.6f} | "
            f"lr={lr:.8f}",
            flush=True,
        )

        save_checkpoint(
            run_dirs["checkpoints"] / "latest.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_dice,
            experiment_config,
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            no_improve = 0

            save_checkpoint(
                run_dirs["checkpoints"] / "best_dice.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_dice,
                experiment_config,
            )

            overlay_dir = run_dirs["overlays"] / f"epoch_{epoch:03d}_best"
            overlay_dir.mkdir(parents=True, exist_ok=True)
            save_val_overlays(model, val_loader, device, overlay_dir, max_samples=6, threshold=args.threshold)
        else:
            no_improve += 1

        if args.smoke_test:
            break

        if no_improve >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.", flush=True)
            break

    summary = {
        "best_dice": round(float(best_dice), 6),
        "best_epoch": best_epoch,
        "total_time_sec": round(float(time.time() - start_all), 3),
        "run_dir": str(run_dirs["run_dir"]),
        "checkpoint_best": str(run_dirs["checkpoints"] / "best_dice.pt"),
        "checkpoint_latest": str(run_dirs["checkpoints"] / "latest.pt"),
    }
    save_json(summary, run_dirs["metrics"] / "train_summary.json")
    print("12. training finished", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
