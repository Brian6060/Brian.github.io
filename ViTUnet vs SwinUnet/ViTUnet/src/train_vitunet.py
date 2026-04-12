"""ViTUNet 训练脚本（基于新版 model.py + dataloader.py）

功能：
1. 训练 ViTUNet
2. 保存 latest / best_dice checkpoint
3. 记录 train / val loss 与 IoU / Dice / AP
4. 保存若干验证集 overlay
5. 支持 smoke test
6. 默认支持 ViT 官方匹配预训练权重（timm 自动加载）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# ---------------------------------------------------------------------
# 让当前脚本可以访问项目根目录与 shared 目录
# ---------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHARED_DIR = _REPO_ROOT / "shared"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SHARED_DIR))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    DATA_ROOT,
    SPLIT_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    NUM_WORKERS,
    THRESHOLD,
    SEED,
    get_device,
    ensure_run_dirs,
)
from losses import BCEDiceLoss
from utils.metrics import compute_metrics, summarize_metrics
from utils.checkpoint import save_checkpoint
from utils.logger import TrainingLogger
from utils.seed import set_seed
from utils.visualize import save_overlay

from dataloader import get_dataloader, KvasirDataset
from model import ViTUNet


# ---------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ViTUNet on Kvasir-SEG")

    parser.add_argument("--data_root", type=Path, default=DATA_ROOT, help="数据集根目录")
    parser.add_argument("--split_dir", type=Path, default=SPLIT_DIR, help="train/val/test split 所在目录")
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=_REPO_ROOT / "runs" / "vitunet",
        help="训练输出目录",
    )

    parser.add_argument("--epochs", type=int, default=EPOCHS, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument("--lr", type=float, default=LR, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="权重衰减")
    parser.add_argument("--device", default="auto", help="设备: auto/cpu/cuda/mps")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="DataLoader worker 数")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子")
    parser.add_argument("--early_stopping_patience", type=int, default=8, help="早停 patience")

    # 新版 ViT 模型相关参数
    parser.add_argument(
        "--backbone",
        type=str,
        default="vit_small_patch16_224",
        help="timm backbone 名称",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="关闭预训练；默认开启官方匹配预训练",
    )

    # 调试相关
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="仅用少量样本快速验证训练链路是否正常",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# 验证函数
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[Dict[str, float], list[Dict[str, Any]]]:
    """在验证集上评估模型。

    返回：
    - summary: 平均 loss / iou / dice / ap
    - records: 每张图的详细指标
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0
    records = []

    for images, masks, meta in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        probs = torch.sigmoid(logits).detach().cpu()
        masks_cpu = masks.detach().cpu()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        for i in range(bs):
            m = compute_metrics(
                probs[i, 0],
                masks_cpu[i, 0],
                threshold=threshold,
            )

            # dataloader 里 meta["filename"] 应该是 list[str]
            filename = (
                meta["filename"][i]
                if isinstance(meta["filename"], list)
                else str(meta["filename"])
            )
            m["filename"] = filename
            records.append(m)

    avg_loss = total_loss / max(total_samples, 1)
    summary = summarize_metrics(records)
    summary["loss"] = round(avg_loss, 6)
    return summary, records


# ---------------------------------------------------------------------
# 单个 epoch 训练
# ---------------------------------------------------------------------
def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
) -> float:
    """训练一个 epoch，返回平均训练 loss。"""
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return round(total_loss / max(total_samples, 1), 6)


# ---------------------------------------------------------------------
# 保存若干验证集 overlay，方便看可视化效果
# ---------------------------------------------------------------------
def save_val_overlays(
    model: torch.nn.Module,
    val_dataset,
    device: torch.device,
    overlay_dir: Path,
    epoch: int,
    n: int = 4,
    threshold: float = 0.5,
) -> None:
    """从验证集随机抽若干张，保存 overlay 图。"""
    import random

    overlay_dir.mkdir(parents=True, exist_ok=True)

    if len(val_dataset) == 0:
        return

    indices = random.sample(range(len(val_dataset)), min(n, len(val_dataset)))
    model.eval()

    with torch.no_grad():
        for idx in indices:
            img_t, msk_t, meta = val_dataset[idx]

            logits = model(img_t.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits)[0, 0].cpu()
            pred = (prob >= threshold).float()

            fname = Path(meta["filename"]).stem
            save_overlay(
                image=img_t,
                gt_mask=msk_t[0],
                pred_mask=pred,
                save_path=overlay_dir / f"epoch{epoch:03d}_{fname}.png",
            )


# ---------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # 1. 固定随机种子
    set_seed(args.seed)

    # 2. 确定设备
    device = get_device(args.device)

    # 3. 创建输出目录
    dirs = ensure_run_dirs(args.save_dir)

    # 4. 保存当前训练配置，方便 test / inference 重建模型
    pretrained = not args.no_pretrained
    cfg = {
        "data_root": str(args.data_root),
        "split_dir": str(args.split_dir),
        "save_dir": str(args.save_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "num_workers": args.num_workers,
        "seed": args.seed,
        "early_stopping_patience": args.early_stopping_patience,
        "backbone": args.backbone,
        "pretrained": pretrained,
        "image_size": IMAGE_SIZE,
        "threshold": THRESHOLD,
        "smoke_test": args.smoke_test,
        "model_name": "ViTUNet",
    }
    (dirs["base"] / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 5. 构建数据
    # smoke_test 只取少量样本，快速看链路是否跑通
    train_limit = 16 if args.smoke_test else None
    val_limit = 8 if args.smoke_test else None

    train_csv = args.split_dir / "train.csv"
    val_csv = args.split_dir / "val.csv"

    train_loader = get_dataloader(
        train_csv,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=train_limit,
    )
    val_loader = get_dataloader(
        val_csv,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=val_limit,
    )
    val_dataset = KvasirDataset(
        split_csv=val_csv,
        split="val",
        limit=val_limit,
    )

    # 6. 构建模型
    model = ViTUNet(
        backbone=args.backbone,
        pretrained=pretrained,
        num_classes=1,
    ).to(device)

    # 7. 优化器 / 调度器 / 损失
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )
    criterion = BCEDiceLoss()

    # 8. 日志器
    logger = TrainingLogger(
        log_dir=dirs["logs"],
        model_name="ViTUNet",
    )

    # 9. 训练状态
    best_dice = -1.0
    patience_counter = 0

    print(
        f"开始训练 ViTUNet | device={device} | epochs={args.epochs} | "
        f"batch_size={args.batch_size} | pretrained={pretrained} | smoke_test={args.smoke_test}"
    )

    for epoch in range(1, args.epochs + 1):
        # ------------------------
        # 训练
        # ------------------------
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        # ------------------------
        # 验证
        # ------------------------
        val_summary, val_records = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=THRESHOLD,
        )

        scheduler.step()
        lr_now = float(optimizer.param_groups[0]["lr"])

        # ------------------------
        # 记录日志
        # ------------------------
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_summary["loss"],
            val_iou=val_summary["iou"],
            val_dice=val_summary["dice"],
            val_ap=val_summary["ap"],
            lr=lr_now,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_summary['loss']:.4f} | "
            f"val_dice={val_summary['dice']:.4f} | "
            f"val_iou={val_summary['iou']:.4f} | "
            f"val_ap={val_summary['ap']:.4f} | "
            f"lr={lr_now:.2e}"
        )

        # ------------------------
        # 保存 overlay
        # ------------------------
        try:
            save_val_overlays(
                model=model,
                val_dataset=val_dataset,
                device=device,
                overlay_dir=dirs["overlays"] / f"epoch_{epoch:03d}",
                epoch=epoch,
                n=4,
                threshold=THRESHOLD,
            )
        except Exception as e:
            print(f"[警告] overlay 保存失败：{e}")

        # ------------------------
        # 保存 latest checkpoint
        # ------------------------
        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_dice": best_dice,
            "val_summary": val_summary,
            "config": cfg,
        }
        save_checkpoint(ckpt_state, dirs["checkpoints"] / "latest.pt")

        # ------------------------
        # 更新 best checkpoint
        # ------------------------
        if val_summary["dice"] > best_dice:
            best_dice = val_summary["dice"]
            patience_counter = 0

            # 这里重新写入 best_dice，确保 best checkpoint 内部信息正确
            ckpt_state["best_dice"] = best_dice
            save_checkpoint(ckpt_state, dirs["checkpoints"] / "best_dice.pt")
        else:
            patience_counter += 1

        # ------------------------
        # 早停
        # ------------------------
        if patience_counter >= args.early_stopping_patience:
            print(f"早停触发：epoch={epoch}，patience={args.early_stopping_patience}")
            break

    # 10. 保存训练总结
    logger.save_summary()

    train_summary = {
        "best_dice": round(float(best_dice), 6),
        "best_epoch": logger.best_epoch if hasattr(logger, "best_epoch") else None,
        "checkpoints_dir": str(dirs["checkpoints"]),
        "logs_dir": str(dirs["logs"]),
        "overlays_dir": str(dirs["overlays"]),
    }
    (dirs["metrics"] / "train_summary.json").write_text(
        json.dumps(train_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n训练完成。")
    print(f"最佳验证集 Dice：{best_dice:.4f}")
    print(f"Checkpoint 目录：{dirs['checkpoints']}")


if __name__ == "__main__":
    main()