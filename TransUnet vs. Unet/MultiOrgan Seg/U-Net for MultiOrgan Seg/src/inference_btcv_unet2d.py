#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


DATA_ROOT = "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/Dataset 2D/processed/BTCV_multiorgan_2d"
CKPT_PATH = "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/U-Net for MultiOrgan Seg/output/train_output/best.pth"
SAVE_ROOT = "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/U-Net for MultiOrgan Seg/output/final_output"
BATCH_SIZE = 8
NUM_WORKERS = 0
INPUT_SIZE = (320, 320)
BASE_CHANNELS = 64
INFERENCE_MODE = "split"
TARGET_SPLIT = "test"
TARGET_CASE_ID = None
IGNORE_BACKGROUND = True
MAX_INFER_STEPS = None
MAX_OVERLAY_EXPORT = 100
MODEL_NAME = "UNet2D_BTCV"


THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from btcv2d_unet_dataloader import BTCV2DUNetDataset, build_btcv2d_unet_dataloader
from unet2d_btcv import UNet2D


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def load_label_map(data_root: str | Path) -> Dict[int, str]:
    label_map_path = Path(data_root) / "meta" / "label_map.json"
    with label_map_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def load_checkpoint_state_dict(checkpoint_path: str | Path, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")


def resolve_total_steps(loader, max_steps: int | None) -> int:
    if max_steps is None:
        return len(loader)
    return min(len(loader), max_steps)


def build_single_case_loader(
    data_root: str | Path,
    case_id: str,
    batch_size: int,
    num_workers: int,
    input_size,
):
    dataset = BTCV2DUNetDataset(
        root_dir=data_root,
        mode="test",
        input_size=input_size,
        samples_per_epoch=1,
        foreground_ratio=0.7,
        use_augmentation=False,
        random_seed=42,
    )
    slices_csv = Path(data_root) / "meta" / "slices.csv"
    all_rows = pd.read_csv(slices_csv).to_dict(orient="records")
    dataset.records = [row for row in all_rows if str(row["case_id"]) == str(case_id)]
    if not dataset.records:
        raise ValueError(f"No slices found for case_id={case_id}")
    dataset.foreground_records = [row for row in dataset.records if int(row["has_organ"]) == 1]
    dataset.background_records = [row for row in dataset.records if int(row["has_organ"]) == 0]
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )


def multiclass_dice_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_background: bool = True,
) -> tuple[float, float]:
    per_class = per_class_dice_iou(preds, targets, num_classes, ignore_background=ignore_background)
    dice_values = [metrics["dice"] for metrics in per_class.values()]
    iou_values = [metrics["iou"] for metrics in per_class.values()]
    if not dice_values:
        return 0.0, 0.0
    return float(np.mean(dice_values)), float(np.mean(iou_values))


def per_class_dice_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_background: bool = True,
) -> Dict[int, Dict[str, float]]:
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    if preds.ndim == 2:
        preds = preds.unsqueeze(0)
    if targets.ndim == 2:
        targets = targets.unsqueeze(0)

    class_range = range(1, num_classes) if ignore_background else range(num_classes)
    per_class: Dict[int, Dict[str, float]] = {}

    for class_id in class_range:
        pred_mask = preds == class_id
        target_mask = targets == class_id
        pred_sum = int(pred_mask.sum().item())
        target_sum = int(target_mask.sum().item())
        if pred_sum == 0 and target_sum == 0:
            continue

        intersection = float((pred_mask & target_mask).sum().item())
        union = float((pred_mask | target_mask).sum().item())
        dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-8)
        iou = intersection / (union + 1e-8)
        per_class[class_id] = {"dice": float(dice), "iou": float(iou)}

    return per_class


def save_prediction(prediction: np.ndarray, save_dir: str | Path, case_id: str, slice_id: str) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{case_id}_{slice_id}_pred.npy"
    np.save(save_path, prediction.astype(np.uint8))
    return save_path


def _color_map(num_classes: int) -> np.ndarray:
    cmap = plt.get_cmap("tab20", max(num_classes, 20))
    return np.array([cmap(i) for i in range(max(num_classes, 20))])


def _present_classes(mask: np.ndarray, ignore_background: bool = True) -> List[int]:
    classes = [int(v) for v in np.unique(mask).tolist()]
    if ignore_background:
        classes = [v for v in classes if v != 0]
    return classes


def save_overlay(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_dir: str | Path,
    case_id: str,
    slice_id: str,
    sample_dice: float,
    sample_iou: float,
    num_classes: int,
) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{case_id}_{slice_id}_overlay.png"

    colors = _color_map(num_classes)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Image", "GT Overlay", "Pred Overlay"]
    masks = [None, gt_mask, pred_mask]

    for axis, title, mask in zip(axes, titles, masks):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        if mask is not None:
            for class_id in _present_classes(mask, ignore_background=IGNORE_BACKGROUND):
                axis.contour(mask == class_id, levels=[0.5], colors=[colors[class_id]], linewidths=1.0)
        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(f"{case_id} | {slice_id} | dice={sample_dice:.4f} | iou={sample_iou:.4f}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def summarize_metrics(
    metrics_per_sample: List[Dict[str, object]],
    per_class_metric_rows: List[Dict[str, object]],
    label_map: Dict[int, str],
) -> tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    if metrics_per_sample:
        mean_dice = float(np.mean([float(row["mean_dice"]) for row in metrics_per_sample]))
        mean_iou = float(np.mean([float(row["mean_iou"]) for row in metrics_per_sample]))
    else:
        mean_dice = 0.0
        mean_iou = 0.0

    grouped: Dict[int, Dict[str, object]] = {}
    for row in per_class_metric_rows:
        class_id = int(row["class_id"])
        grouped.setdefault(class_id, {"dice": [], "iou": [], "class_name": row["class_name"]})
        grouped[class_id]["dice"].append(float(row["dice"]))
        grouped[class_id]["iou"].append(float(row["iou"]))

    metrics_per_class: List[Dict[str, object]] = []
    per_class_summary: Dict[str, Dict[str, object]] = {}
    final_summary_row: Dict[str, object] = {
        "model_name": MODEL_NAME,
        "checkpoint_path": CKPT_PATH,
        "num_test_samples": len(metrics_per_sample),
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
    }

    for class_id in sorted(label_map.keys()):
        if IGNORE_BACKGROUND and class_id == 0:
            continue
        class_name = label_map[class_id]
        values = grouped.get(class_id, {"dice": [], "iou": [], "class_name": class_name})
        dice_values = values["dice"]
        iou_values = values["iou"]
        class_mean_dice = float(np.mean(dice_values)) if dice_values else 0.0
        class_mean_iou = float(np.mean(iou_values)) if iou_values else 0.0
        support_samples = len(dice_values)
        row = {
            "class_id": class_id,
            "class_name": class_name,
            "mean_dice": class_mean_dice,
            "mean_iou": class_mean_iou,
            "support_samples": support_samples,
        }
        metrics_per_class.append(row)
        per_class_summary[str(class_id)] = row
        final_summary_row[f"class_{class_id}_dice"] = class_mean_dice
        final_summary_row[f"class_{class_id}_iou"] = class_mean_iou

    final_summary_json = {
        "model_name": MODEL_NAME,
        "checkpoint_path": CKPT_PATH,
        "num_test_samples": len(metrics_per_sample),
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "input_size": list(INPUT_SIZE),
        "ignore_background": IGNORE_BACKGROUND,
        "per_class_summary": per_class_summary,
    }
    return metrics_per_class, final_summary_json, final_summary_row


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    label_map: Dict[int, str],
    save_root: str | Path,
) -> None:
    save_root = Path(save_root)
    predictions_dir = save_root / "predictions"
    overlays_dir = save_root / "overlays"
    save_root.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    inference_rows: List[Dict[str, object]] = []
    metrics_per_sample: List[Dict[str, object]] = []
    per_class_metric_rows: List[Dict[str, object]] = []

    total_steps = resolve_total_steps(loader, MAX_INFER_STEPS)
    progress = tqdm(range(total_steps), total=total_steps, desc="final infer", leave=False)
    iterator = iter(loader)
    exported_overlays = 0

    for batch_index in progress:
        batch = next(iterator)
        images = batch["image"].to(device=device, dtype=torch.float32)
        masks = batch["mask"].to(device=device, dtype=torch.long)
        case_ids = list(batch["case_id"])
        slice_ids = list(batch["slice_id"])

        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        batch_dice, batch_iou = multiclass_dice_iou(
            preds, masks, num_classes=num_classes, ignore_background=IGNORE_BACKGROUND
        )
        progress.set_postfix(dice=f"{batch_dice:.4f}", iou=f"{batch_iou:.4f}")

        for local_idx, (case_id, slice_id) in enumerate(zip(case_ids, slice_ids)):
            image_np = images[local_idx, 0].detach().cpu().numpy()
            gt_np = masks[local_idx].detach().cpu().numpy()
            pred_np = preds[local_idx].detach().cpu().numpy()

            sample_per_class = per_class_dice_iou(
                preds[local_idx],
                masks[local_idx],
                num_classes=num_classes,
                ignore_background=IGNORE_BACKGROUND,
            )
            sample_dice, sample_iou = multiclass_dice_iou(
                preds[local_idx],
                masks[local_idx],
                num_classes=num_classes,
                ignore_background=IGNORE_BACKGROUND,
            )

            pred_path = save_prediction(pred_np, predictions_dir, str(case_id), str(slice_id))
            overlay_path = ""
            if exported_overlays < MAX_OVERLAY_EXPORT:
                overlay = save_overlay(
                    image=image_np,
                    gt_mask=gt_np,
                    pred_mask=pred_np,
                    save_dir=overlays_dir,
                    case_id=str(case_id),
                    slice_id=str(slice_id),
                    sample_dice=sample_dice,
                    sample_iou=sample_iou,
                    num_classes=num_classes,
                )
                overlay_path = str(overlay)
                exported_overlays += 1

            inference_rows.append(
                {
                    "index": len(inference_rows),
                    "case_id": str(case_id),
                    "slice_id": str(slice_id),
                    "pred_path": str(pred_path),
                    "overlay_path": overlay_path,
                }
            )
            metrics_per_sample.append(
                {
                    "index": len(metrics_per_sample),
                    "case_id": str(case_id),
                    "slice_id": str(slice_id),
                    "mean_dice": sample_dice,
                    "mean_iou": sample_iou,
                    "present_gt_classes": ",".join(map(str, _present_classes(gt_np, IGNORE_BACKGROUND))),
                    "present_pred_classes": ",".join(map(str, _present_classes(pred_np, IGNORE_BACKGROUND))),
                }
            )

            for class_id, metrics in sample_per_class.items():
                per_class_metric_rows.append(
                    {
                        "index": len(metrics_per_sample) - 1,
                        "case_id": str(case_id),
                        "slice_id": str(slice_id),
                        "class_id": class_id,
                        "class_name": label_map.get(class_id, f"class_{class_id}"),
                        "dice": metrics["dice"],
                        "iou": metrics["iou"],
                    }
                )

    metrics_per_class, final_summary_json, final_summary_row = summarize_metrics(
        metrics_per_sample=metrics_per_sample,
        per_class_metric_rows=per_class_metric_rows,
        label_map=label_map,
    )

    with (save_root / "inference_records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "case_id", "slice_id", "pred_path", "overlay_path"])
        writer.writeheader()
        writer.writerows(inference_rows)

    with (save_root / "metrics_per_sample.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "case_id",
                "slice_id",
                "mean_dice",
                "mean_iou",
                "present_gt_classes",
                "present_pred_classes",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_per_sample)

    with (save_root / "metrics_per_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "class_name", "mean_dice", "mean_iou", "support_samples"],
        )
        writer.writeheader()
        writer.writerows(metrics_per_class)

    with (save_root / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(final_summary_json, f, indent=2, ensure_ascii=False)

    with (save_root / "final_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_summary_row.keys()))
        writer.writeheader()
        writer.writerow(final_summary_row)

    print(f"num_test_samples: {final_summary_json['num_test_samples']}")
    print(f"mean_dice: {final_summary_json['mean_dice']:.6f}")
    print(f"mean_iou: {final_summary_json['mean_iou']:.6f}")


def main() -> None:
    data_root = Path(DATA_ROOT)
    checkpoint_path = Path(CKPT_PATH)
    save_root = Path(SAVE_ROOT)
    save_root.mkdir(parents=True, exist_ok=True)
    (save_root / "predictions").mkdir(parents=True, exist_ok=True)
    (save_root / "overlays").mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    label_map = load_label_map(data_root)
    num_classes = max(label_map.keys()) + 1

    if INFERENCE_MODE == "split":
        loader = build_btcv2d_unet_dataloader(
            root_dir=data_root,
            mode=TARGET_SPLIT,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            input_size=INPUT_SIZE,
            foreground_ratio=0.7,
            use_augmentation=False,
            random_seed=42,
            pin_memory=(DEVICE.type == "cuda"),
            drop_last=False,
        )
        target_desc = TARGET_SPLIT
    elif INFERENCE_MODE == "single_case":
        if TARGET_CASE_ID is None:
            raise ValueError("TARGET_CASE_ID must be provided for single_case mode.")
        loader = build_single_case_loader(
            data_root=data_root,
            case_id=str(TARGET_CASE_ID),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            input_size=INPUT_SIZE,
        )
        target_desc = str(TARGET_CASE_ID)
    else:
        raise ValueError("INFERENCE_MODE must be 'split' or 'single_case'.")

    model = UNet2D(in_channels=1, num_classes=num_classes, base_channels=BASE_CHANNELS).to(DEVICE)
    model.load_state_dict(load_checkpoint_state_dict(checkpoint_path, DEVICE))

    print(f"device: {DEVICE}")
    print(f"data_root: {data_root}")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"save_root: {save_root}")
    print(f"model_name: {MODEL_NAME}")
    print(f"num_classes: {num_classes}")
    print(f"input_size: {INPUT_SIZE}")
    print(f"inference_mode: {INFERENCE_MODE}")
    if INFERENCE_MODE == "split":
        print(f"target_split: {target_desc}")
    else:
        print(f"target_case_id: {target_desc}")
    print(f"test sample count: {len(loader.dataset)}")

    run_inference(model, loader, DEVICE, num_classes, label_map, save_root)


if __name__ == "__main__":
    main()
