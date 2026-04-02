#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from DataLoader2_0 import build_unet_strict_train_loader
from Model import UNetPaper


DEFAULT_PROCESSED_ROOT = Path(
    "/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0/processed_unet_train_auto_originalstyle"
)
DEFAULT_SAVE_DIR = Path(
    "/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0/outputs_train_improved"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train paper-style U-Net with train-only workflow.")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--input-size", type=int, default=572)
    parser.add_argument("--output-size", type=int, default=388)
    parser.add_argument("--patches-per-image", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--elastic-deform", action="store_true", default=True)
    parser.add_argument("--no-elastic-deform", dest="elastic_deform", action="store_false")
    parser.add_argument("--displacement-std", type=float, default=10.0)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--normalize", type=str, default="zscore", choices=["zscore", "minmax", "none"])
    parser.add_argument("--gray-value-aug", action="store_true", default=True)
    parser.add_argument("--use-bottleneck-dropout", action="store_true", default=True)
    parser.add_argument("--no-bottleneck-dropout", dest="use_bottleneck_dropout", action="store_false")
    parser.add_argument("--dropout-p", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS is not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def weighted_pixelwise_cross_entropy(logits: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"logits shape must be [B,C,H,W], got {tuple(logits.shape)}")
    if target.ndim != 3:
        raise ValueError(f"target shape must be [B,H,W], got {tuple(target.shape)}")
    if weight_map.ndim != 3:
        raise ValueError(f"weight_map shape must be [B,H,W], got {tuple(weight_map.shape)}")
    log_probs = F.log_softmax(logits, dim=1)
    picked = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    return -(weight_map * picked).sum() / weight_map.sum().clamp_min(1e-6)


def _binary_confusion_from_logits(logits: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float, float]:
    pred = logits.argmax(dim=1)
    pred_pos = pred == 1
    true_pos = target == 1
    tp = torch.logical_and(pred_pos, true_pos).sum().item()
    fp = torch.logical_and(pred_pos, ~true_pos).sum().item()
    fn = torch.logical_and(~pred_pos, true_pos).sum().item()
    tn = torch.logical_and(~pred_pos, ~true_pos).sum().item()
    return float(tp), float(fp), float(fn), float(tn)


def dice_score_binary(tp: float, fp: float, fn: float, eps: float = 1e-6) -> float:
    return float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps))


def iou_score_binary(tp: float, fp: float, fn: float, eps: float = 1e-6) -> float:
    return float((tp + eps) / (tp + fp + fn + eps))


def save_checkpoint(
    save_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_loss: float,
    args: argparse.Namespace,
    metrics: Dict[str, float],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "metrics": metrics,
        "args": vars(args),
    }
    torch.save(ckpt, str(save_path))


def load_checkpoint(ckpt_path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    best_loss = float(ckpt.get("best_loss", float("inf")))
    return epoch, global_step, best_loss


def build_train_loader(args: argparse.Namespace):
    normalize = None if args.normalize == "none" else args.normalize
    return build_unet_strict_train_loader(
        processed_root=args.processed_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        input_size=args.input_size,
        output_size=args.output_size,
        patches_per_image=args.patches_per_image,
        elastic_deform=args.elastic_deform,
        displacement_std=args.displacement_std,
        grid_size=args.grid_size,
        normalize=normalize,
        gray_value_aug=args.gray_value_aug,
        w0=10.0,
        sigma=5.0,
    )


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int,
    global_step: int,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    num_batches = 0
    tp = fp = fn = tn = 0.0
    start = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        weight = batch["weight"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = weighted_pixelwise_cross_entropy(logits, target, weight)
        loss.backward()
        optimizer.step()

        batch_tp, batch_fp, batch_fn, batch_tn = _binary_confusion_from_logits(logits.detach(), target)
        batch_dice = dice_score_binary(batch_tp, batch_fp, batch_fn)
        batch_iou = iou_score_binary(batch_tp, batch_fp, batch_fn)
        tp += batch_tp
        fp += batch_fp
        fn += batch_fn
        tn += batch_tn

        running_loss += float(loss.item())
        num_batches += 1
        global_step += 1

        if batch_idx % print_freq == 0 or batch_idx == 1 or batch_idx == len(loader):
            print(
                f"[Epoch {epoch:03d}] "
                f"[Iter {batch_idx:04d}/{len(loader):04d}] "
                f"loss={loss.item():.6f} "
                f"dice={batch_dice:.4f} "
                f"iou={batch_iou:.4f}"
            )

    return {
        "epoch_loss": running_loss / max(num_batches, 1),
        "epoch_dice": dice_score_binary(tp, fp, fn),
        "epoch_iou": iou_score_binary(tp, fp, fn),
        "epoch_time_sec": time.time() - start,
        "global_step": global_step,
    }


def save_curves(records: List[Dict[str, float]], save_dir: Path) -> None:
    if not records:
        return

    epochs = [row["epoch"] for row in records]
    losses = [row["loss"] for row in records]
    dices = [row["dice"] for row in records]
    ious = [row["iou"] for row in records]

    def _save_curve(path: Path, ylabel: str, values: List[float]) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, values, marker="o", label=ylabel.lower())
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} Curve")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    _save_curve(save_dir / "loss_curve.png", "Loss", losses)
    _save_curve(save_dir / "dice_curve.png", "Dice", dices)
    _save_curve(save_dir / "iou_curve.png", "IoU", ious)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    args.processed_root = args.processed_root.expanduser().resolve()
    args.save_dir = args.save_dir.expanduser().resolve()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print("========== Train Config ==========")
    print(json.dumps({
        "processed_root": str(args.processed_root),
        "save_dir": str(args.save_dir),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "input_size": args.input_size,
        "output_size": args.output_size,
        "patches_per_image": args.patches_per_image,
        "elastic_deform": args.elastic_deform,
        "displacement_std": args.displacement_std,
        "grid_size": args.grid_size,
        "normalize": args.normalize,
        "gray_value_aug": args.gray_value_aug,
        "use_bottleneck_dropout": args.use_bottleneck_dropout,
        "dropout_p": args.dropout_p,
    }, indent=2, ensure_ascii=False))
    print("==================================")

    loader = build_train_loader(args)
    model = UNetPaper(
        in_channels=1,
        num_classes=2,
        use_bottleneck_dropout=args.use_bottleneck_dropout,
        bottleneck_dropout_p=args.dropout_p,
    ).to(device)
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    global_step = 0
    best_loss = float("inf")

    if args.resume:
        ckpt_path = Path(args.resume).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        last_epoch, global_step, best_loss = load_checkpoint(ckpt_path, model, optimizer, device)
        start_epoch = last_epoch + 1
        print(f"Resumed from: {ckpt_path}")
        print(f"Start epoch: {start_epoch}, global_step: {global_step}, best_loss: {best_loss:.6f}")

    log_csv = args.save_dir / "train_log.csv"
    records: List[Dict[str, float]] = []
    if log_csv.exists():
        log_csv.unlink()

    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "dice", "iou", "time_sec", "global_step"])

        for epoch in range(start_epoch, args.epochs + 1):
            stats = train_one_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                print_freq=args.print_freq,
                global_step=global_step,
            )
            global_step = int(stats["global_step"])
            epoch_loss = float(stats["epoch_loss"])
            epoch_dice = float(stats["epoch_dice"])
            epoch_iou = float(stats["epoch_iou"])
            epoch_time = float(stats["epoch_time_sec"])

            print(
                f">>> Epoch {epoch:03d} done: "
                f"loss={epoch_loss:.6f}, "
                f"dice={epoch_dice:.4f}, "
                f"iou={epoch_iou:.4f}, "
                f"time={epoch_time:.2f}s"
            )

            writer.writerow([epoch, epoch_loss, epoch_dice, epoch_iou, epoch_time, global_step])
            f.flush()
            records.append({
                "epoch": epoch,
                "loss": epoch_loss,
                "dice": epoch_dice,
                "iou": epoch_iou,
                "time_sec": epoch_time,
                "global_step": global_step,
            })

            metrics = {
                "loss": epoch_loss,
                "dice": epoch_dice,
                "iou": epoch_iou,
                "time_sec": epoch_time,
                "global_step": global_step,
            }

            latest_path = args.save_dir / "latest.pt"
            save_checkpoint(latest_path, model, optimizer, epoch, global_step, best_loss, args, metrics)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_path = args.save_dir / "best_train_loss.pt"
                save_checkpoint(best_path, model, optimizer, epoch, global_step, best_loss, args, metrics)
                print(f"*** New best train loss: {best_loss:.6f}")

            if epoch % args.save_every == 0:
                epoch_path = args.save_dir / f"epoch_{epoch:03d}.pt"
                save_checkpoint(epoch_path, model, optimizer, epoch, global_step, best_loss, args, metrics)

    save_curves(records, args.save_dir)

    print("Training finished.")
    print(f"Best train loss: {best_loss:.6f}")
    print(f"Logs saved to: {log_csv}")


if __name__ == "__main__":
    main()
