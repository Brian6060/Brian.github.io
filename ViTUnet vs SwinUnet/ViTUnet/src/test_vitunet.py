from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

import torch
from sklearn.metrics import average_precision_score

from dataloader import build_vitunet_dataloader
from model import ViTUNet


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def binary_iou(prob: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    pred = (prob >= threshold).astype(np.uint8)
    gt = (target >= 0.5).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(_safe_div(float(inter), float(union)))


def binary_dice(prob: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    pred = (prob >= threshold).astype(np.uint8)
    gt = (target >= 0.5).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float(_safe_div(float(2 * inter), float(denom)))


def binary_ap(prob: np.ndarray, target: np.ndarray) -> float:
    y_true = target.astype(np.uint8).reshape(-1)
    y_score = prob.astype(np.float32).reshape(-1)
    if np.unique(y_true).size == 1:
        if y_true[0] == 0:
            return 1.0 if float(y_score.max()) < 0.5 else 0.0
        return 1.0
    return float(average_precision_score(y_true, y_score))


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
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


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    model = ViTUNet(
        backbone=config["backbone"],
        pretrained=False,
        num_classes=1,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, config, ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test ViTUNet on Kvasir-SEG")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    model, config, ckpt = load_checkpoint_model(checkpoint_path, device)

    processed_root = Path(config["processed_root"]).expanduser().resolve()
    run_dir = checkpoint_path.parent.parent
    metrics_dir = run_dir / "metrics"
    overlay_dir = run_dir / "test_overlays"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    test_loader = build_vitunet_dataloader(
        processed_root=processed_root,
        split="test",
        image_size=config["image_size"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        use_bbox=True,
    )

    per_image_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        seen = 0
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            metas = batch["meta"]

            logits = model(images)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            gts = masks.detach().cpu().numpy()

            for i in range(images.size(0)):
                stem = metas[i]["stem"]
                prob = probs[i, 0]
                gt = gts[i, 0]

                iou = binary_iou(prob, gt, args.threshold)
                dice = binary_dice(prob, gt, args.threshold)
                ap = binary_ap(prob, gt)

                per_image_rows.append(
                    {
                        "stem": stem,
                        "iou": round(iou, 6),
                        "dice": round(dice, 6),
                        "ap": round(ap, 6),
                    }
                )

                image_rgb = denormalize_image(images[i])
                gt_mask = gt >= 0.5
                pred_mask = prob >= args.threshold
                overlay = make_overlay(image_rgb, gt_mask, pred_mask)

                Image.fromarray(image_rgb).save(overlay_dir / f"{stem}_image.png")
                Image.fromarray((gt_mask.astype(np.uint8) * 255)).save(overlay_dir / f"{stem}_gt.png")
                Image.fromarray((pred_mask.astype(np.uint8) * 255)).save(overlay_dir / f"{stem}_pred.png")
                Image.fromarray(overlay).save(overlay_dir / f"{stem}_overlay.png")

                seen += 1
                if args.smoke_test and seen >= 4:
                    break

            if args.smoke_test and seen >= 4:
                break

    mean_iou = float(np.mean([r["iou"] for r in per_image_rows])) if per_image_rows else 0.0
    mean_dice = float(np.mean([r["dice"] for r in per_image_rows])) if per_image_rows else 0.0
    mean_ap = float(np.mean([r["ap"] for r in per_image_rows])) if per_image_rows else 0.0

    summary = {
        "checkpoint": str(checkpoint_path),
        "epoch": ckpt.get("epoch", -1),
        "best_dice_recorded": ckpt.get("best_dice", None),
        "num_test_images": len(per_image_rows),
        "mean_iou": round(mean_iou, 6),
        "mean_dice": round(mean_dice, 6),
        "mean_ap": round(mean_ap, 6),
        "threshold": args.threshold,
        "smoke_test": args.smoke_test,
    }

    with (metrics_dir / "test_metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = metrics_dir / "test_metrics_per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "iou", "dice", "ap"])
        writer.writeheader()
        writer.writerows(per_image_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
