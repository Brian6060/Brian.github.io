from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

import torch
from sklearn.metrics import average_precision_score
from torchvision import transforms

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


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
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
    return model, config


def build_preprocess(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_overlay(image_rgb: np.ndarray, gt_mask: Optional[np.ndarray], pred_mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    overlay = image_rgb.copy()

    if gt_mask is not None:
        gt_region = gt_mask.astype(bool)
        overlay[gt_region] = (
            (1 - alpha) * overlay[gt_region] + alpha * np.array([0, 255, 0], dtype=np.float32)
        ).astype(np.uint8)

    pred_region = pred_mask.astype(bool)
    overlay[pred_region] = (
        (1 - alpha) * overlay[pred_region] + alpha * np.array([255, 0, 0], dtype=np.float32)
    ).astype(np.uint8)

    return overlay


def collect_image_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in exts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for ViTUNet")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = get_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_out = output_dir / "images"
    prob_out = output_dir / "prob_maps"
    pred_out = output_dir / "pred_masks"
    overlay_out = output_dir / "overlays"
    metrics_out = output_dir / "metrics"
    for p in [image_out, prob_out, pred_out, overlay_out, metrics_out]:
        p.mkdir(parents=True, exist_ok=True)

    model, config = load_checkpoint_model(checkpoint_path, device)
    preprocess = build_preprocess(config["image_size"])

    mask_dir = Path(args.mask_dir).expanduser().resolve() if args.mask_dir else None
    image_paths = collect_image_paths(input_path)

    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for image_path in image_paths:
            stem = image_path.stem

            pil_img = Image.open(image_path).convert("RGB")
            resized_img = pil_img.resize((config["image_size"], config["image_size"]), Image.BILINEAR)

            img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            logits = model(img_tensor)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            pred_mask = (prob >= args.threshold).astype(np.uint8)

            image_rgb = np.asarray(resized_img, dtype=np.uint8)

            gt_mask = None
            metrics = {"iou": None, "dice": None, "ap": None}

            if mask_dir is not None:
                gt_candidates = list(mask_dir.glob(f"{stem}.*"))
                if len(gt_candidates) == 1:
                    gt_pil = Image.open(gt_candidates[0]).convert("L")
                    gt_pil = gt_pil.resize((config["image_size"], config["image_size"]), Image.NEAREST)
                    gt_mask = (np.asarray(gt_pil, dtype=np.uint8) > 0).astype(np.uint8)

                    metrics["iou"] = round(binary_iou(prob, gt_mask, args.threshold), 6)
                    metrics["dice"] = round(binary_dice(prob, gt_mask, args.threshold), 6)
                    metrics["ap"] = round(binary_ap(prob, gt_mask), 6)

            overlay = make_overlay(image_rgb, gt_mask, pred_mask)

            Image.fromarray(image_rgb).save(image_out / f"{stem}_image.png")
            Image.fromarray((prob * 255).astype(np.uint8)).save(prob_out / f"{stem}_prob.png")
            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(pred_out / f"{stem}_pred.png")
            Image.fromarray(overlay).save(overlay_out / f"{stem}_overlay.png")

            rows.append(
                {
                    "stem": stem,
                    "image_path": str(image_path),
                    "iou": metrics["iou"],
                    "dice": metrics["dice"],
                    "ap": metrics["ap"],
                }
            )

    csv_path = metrics_out / "inference_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "image_path", "iou", "dice", "ap"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(checkpoint_path),
        "input_path": str(input_path),
        "num_images": len(rows),
        "mask_dir": str(mask_dir) if mask_dir is not None else None,
        "threshold": args.threshold,
    }
    with (metrics_out / "inference_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()