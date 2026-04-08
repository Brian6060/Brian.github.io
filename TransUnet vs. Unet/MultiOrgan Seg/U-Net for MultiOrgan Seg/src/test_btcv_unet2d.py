#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


DATA_ROOT = "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/Dataset 2D/processed/BTCV_multiorgan_2d"
CKPT_PATH = "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/U-Net for MultiOrgan Seg/output/train_output/best.pth"
SAVE_ROOT = "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/U-Net for MultiOrgan Seg/output/test_output"
BATCH_SIZE = 8
NUM_WORKERS = 0
INPUT_SIZE = (320, 320)
BASE_CHANNELS = 64
IGNORE_BACKGROUND = True
MAX_TEST_STEPS = None


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


def load_checkpoint_state_dict(checkpoint_path: str | Path, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")


def load_label_map(data_root: str | Path) -> Dict[int, str]:
    with (Path(data_root) / "meta" / "label_map.json").open("r", encoding="utf-8") as f:
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


def save_prediction(prediction: np.ndarray, save_dir: str | Path, case_id: str, slice_id: str) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{case_id}_{slice_id}.npy"
    np.save(save_path, prediction.astype(np.uint8))
    return save_path


def resolve_total_steps(loader, max_steps: int | None) -> int:
    if max_steps is None:
        return len(loader)
    return min(len(loader), max_steps)


@torch.no_grad()
def run_test(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    num_classes: int,
    save_root: str | Path,
    checkpoint_path: str | Path,
) -> Tuple[float, float]:
    save_root = Path(save_root)
    predictions_dir = save_root / "predictions"
    save_root.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    pred_case_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    dice_scores: List[float] = []
    iou_scores: List[float] = []

    processed_batches = 0
    total_steps = resolve_total_steps(test_loader, MAX_TEST_STEPS)
    progress = tqdm(range(total_steps), total=total_steps, desc="test", leave=False)
    iterator = iter(test_loader)

    for batch_index in progress:
        batch = next(iterator)
        processed_batches += 1
        images = batch["image"].to(device=device, dtype=torch.float32)
        masks = batch["mask"].to(device=device, dtype=torch.long)
        case_ids = list(batch["case_id"])
        slice_ids = list(batch["slice_id"])

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        batch_dice, batch_iou = multiclass_dice_iou(
            preds,
            masks,
            num_classes=num_classes,
            ignore_background=IGNORE_BACKGROUND,
        )
        progress.set_postfix(dice=f"{batch_dice:.4f}", iou=f"{batch_iou:.4f}")

        for local_idx, (case_id, slice_id) in enumerate(zip(case_ids, slice_ids)):
            pred_np = preds[local_idx].detach().cpu().numpy()
            dice, iou = multiclass_dice_iou(
                preds[local_idx],
                masks[local_idx],
                num_classes=num_classes,
                ignore_background=IGNORE_BACKGROUND,
            )
            pred_path = save_prediction(pred_np, predictions_dir, str(case_id), str(slice_id))
            pred_case_rows.append(
                {
                    "batch_index": batch_index,
                    "case_id": str(case_id),
                    "slice_id": str(slice_id),
                    "pred_path": str(pred_path),
                }
            )
            metric_rows.append(
                {
                    "batch_index": batch_index,
                    "case_id": str(case_id),
                    "slice_id": str(slice_id),
                    "dice": f"{dice:.6f}",
                    "iou": f"{iou:.6f}",
                }
            )
            dice_scores.append(dice)
            iou_scores.append(iou)

    with (save_root / "pred_cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_index", "case_id", "slice_id", "pred_path"])
        writer.writeheader()
        writer.writerows(pred_case_rows)

    with (save_root / "test_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_index", "case_id", "slice_id", "dice", "iou"])
        writer.writeheader()
        writer.writerows(metric_rows)

    test_mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    test_mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "num_test_batches": int(processed_batches),
        "test_mean_dice": test_mean_dice,
        "test_mean_iou": test_mean_iou,
        "input_size": list(INPUT_SIZE),
    }
    with (save_root / "test_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"test_mean_dice: {test_mean_dice:.6f}")
    print(f"test_mean_iou: {test_mean_iou:.6f}")
    return test_mean_dice, test_mean_iou


def _case_count(loader) -> int:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return 0
    records = getattr(dataset, "records", [])
    return len({str(row["case_id"]) for row in records})


def main() -> None:
    data_root = Path(DATA_ROOT)
    checkpoint_path = Path(CKPT_PATH)
    save_root = Path(SAVE_ROOT)
    save_root.mkdir(parents=True, exist_ok=True)
    (save_root / "predictions").mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    label_map = load_label_map(data_root)
    num_classes = max(label_map.keys()) + 1

    test_loader = build_btcv2d_unet_dataloader(
        root_dir=data_root,
        mode="test",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        input_size=INPUT_SIZE,
        foreground_ratio=0.7,
        use_augmentation=False,
        random_seed=42,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )

    UNet2D = load_unet2d_class()
    model = UNet2D(in_channels=1, num_classes=num_classes, base_channels=BASE_CHANNELS).to(DEVICE)
    model.load_state_dict(load_checkpoint_state_dict(checkpoint_path, DEVICE))

    print(f"device: {DEVICE}")
    print(f"data_root: {data_root}")
    print(f"num_classes: {num_classes}")
    print(f"checkpoint path: {checkpoint_path}")
    print(f"test case count: {_case_count(test_loader)}")
    print(f"input_size: {INPUT_SIZE}")

    run_test(model, test_loader, DEVICE, num_classes, save_root, checkpoint_path)


if __name__ == "__main__":
    main()
