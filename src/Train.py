#!/usr/bin/env python3
from __future__ import annotations

# ==========
# Train.py
# 这一部分属于复现框架里的 train
# 目标：
# 1. 从 DataLoader 里取出训练 batch
# 2. 把 batch 喂给 U-Net 模型
# 3. 计算论文要求的 pixel-wise weighted cross entropy
# 4. 用 SGD + momentum 更新参数
# 5. 保存日志和 checkpoint
# ==========

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD

# 从你前面已经写好的两个模块中导入：
# DataLoader: 负责生成训练 batch
# Model: 负责前向传播，输出 logits
from DataLoader import build_unet_strict_train_loader
from Model import UNetPaper


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    这些参数分成几类：
    1. 数据路径相关
    2. 训练超参数相关
    3. 数据增强相关
    4. dropout 相关
    5. 设备与保存相关
    """
    parser = argparse.ArgumentParser(description="Train U-Net paper-style in PyTorch.")

    # 处理后的数据根目录
    # 这里应该对应你前面 processed_unet_strict 的输出目录
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0/processed_unet_strict"),
        help="Processed dataset root.",
    )

    # 训练日志和 checkpoint 的保存目录
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0/outputs_train"),
        help="Checkpoint and log output directory.",
    )

    # ==========
    # 原论文核心训练设定
    # ==========
    # batch size = 1
    parser.add_argument("--batch-size", type=int, default=1, help="Paper uses batch size 1.")

    # 高动量 0.99
    parser.add_argument("--momentum", type=float, default=0.99, help="Paper uses high momentum 0.99.")

    # 输入 patch 大小
    # 对应 valid conv 的输入尺寸，比如经典 572
    parser.add_argument("--input-size", type=int, default=572)

    # 输出 patch 大小
    # 对应 U-Net 中心有效预测区域，比如经典 388
    parser.add_argument("--output-size", type=int, default=388)

    # 每张原图每个 epoch 采多少个训练 patch
    parser.add_argument("--patches-per-image", type=int, default=32)

    # ==========
    # 这些是训练超参数
    # 论文没有把它们写成唯一固定值，所以做成可调参数
    # ==========
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    # ==========
    # augmentation / dataset 相关
    # ==========
    # 是否启用 elastic deformation
    parser.add_argument("--elastic-deform", action="store_true", default=True)
    parser.add_argument("--no-elastic-deform", dest="elastic_deform", action="store_false")

    # 位移标准差，论文里核心值是 10 px
    parser.add_argument("--displacement-std", type=float, default=10.0)

    # 粗网格大小，论文里是 3x3
    parser.add_argument("--grid-size", type=int, default=3)

    # 输入图像归一化方式
    parser.add_argument("--normalize", type=str, default="zscore", choices=["zscore", "minmax", "none"])

    # 是否启用灰度扰动
    # 论文提到 gray value variations，但没写死唯一公式
    parser.add_argument("--gray-value-aug", action="store_true", default=True)

    # ==========
    # dropout 相关
    # ==========
    parser.add_argument("--use-bottleneck-dropout", action="store_true", default=True)
    parser.add_argument("--no-bottleneck-dropout", dest="use_bottleneck_dropout", action="store_false")
    parser.add_argument("--dropout-p", type=float, default=0.5)

    # ==========
    # 其他工程参数
    # ==========
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="")

    # 设备选择
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """
    固定随机种子，尽量提高实验可复现性。

    为什么要做：
    1. Python random 会用到
    2. NumPy 会用到
    3. PyTorch 会用到
    4. 如果有 CUDA，也要固定 CUDA 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    """
    根据命令行参数选择训练设备。

    支持：
    1. cpu
    2. cuda
    3. mps
    4. auto 自动选择
    """
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

    # auto 模式下优先 CUDA，再 MPS，最后 CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def weighted_pixelwise_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor,
) -> torch.Tensor:
    """
    论文对应的核心 loss：
    pixel-wise softmax + weighted cross entropy

    输入：
    logits:     [B, C, H, W]
                模型输出的未归一化分数
    target:     [B, H, W]
                每个像素的类别标签，0 或 1
    weight_map: [B, H, W]
                每个像素自己的权重

    返回：
    一个标量 loss

    这里为什么自己手写：
    因为论文用的是逐像素权重图，不是简单的类别权重。
    PyTorch 自带 CrossEntropyLoss 的 class weight 不够表达论文这个形式。
    """

    # 先做输入 shape 检查，防止 silent bug
    if logits.ndim != 4:
        raise ValueError(f"logits shape must be [B,C,H,W], got {tuple(logits.shape)}")
    if target.ndim != 3:
        raise ValueError(f"target shape must be [B,H,W], got {tuple(target.shape)}")
    if weight_map.ndim != 3:
        raise ValueError(f"weight_map shape must be [B,H,W], got {tuple(weight_map.shape)}")

    # 对类别维做 log_softmax
    # 输出还是 [B, C, H, W]
    log_probs = F.log_softmax(logits, dim=1)

    # 从 log_probs 中，取出每个像素对应真实类别的 log probability
    # target.unsqueeze(1): [B,1,H,W]
    # gather 后变成 [B,1,H,W]
    # squeeze(1) 后变成 [B,H,W]
    picked = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)

    # 负对数似然，再乘以逐像素权重
    # 最后用 weight_map.sum() 做归一化
    loss = -(weight_map * picked).sum() / weight_map.sum().clamp_min(1e-6)
    return loss


@torch.no_grad()
def pixel_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算 pixel accuracy。

    这里只是一个简单训练监控指标，不是最终论文级评估指标。
    用途：
    1. 看训练有没有基本学起来
    2. 快速发现模型输出全黑/全白等明显问题
    """
    pred = logits.argmax(dim=1)
    acc = (pred == target).float().mean().item()
    return float(acc)


def save_checkpoint(
    save_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_loss: float,
    args: argparse.Namespace,
) -> None:
    """
    保存 checkpoint。

    保存内容包括：
    1. 模型参数
    2. 优化器状态
    3. 当前 epoch
    4. 全局 step
    5. 当前 best loss
    6. 训练参数配置
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "args": vars(args),
    }

    torch.save(ckpt, str(save_path))


def load_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """
    从 checkpoint 恢复训练状态。

    返回：
    1. epoch
    2. global_step
    3. best_loss
    """
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    best_loss = float(ckpt.get("best_loss", float("inf")))

    return epoch, global_step, best_loss


def build_train_loader(args: argparse.Namespace):
    """
    根据参数构建训练 DataLoader。

    这里本质上是把 Train.py 里的参数传给 DataLoader.py。
    也就是把 train 配置和 dataloader 配置接起来。
    """
    normalize = None if args.normalize == "none" else args.normalize

    loader = build_unet_strict_train_loader(
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
    return loader


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int,
    global_step: int,
) -> Dict[str, float]:
    """
    训练一个 epoch。

    一个 epoch 内部做的事情：
    1. 从 DataLoader 取 batch
    2. 前向传播
    3. 计算 weighted CE loss
    4. 反向传播
    5. SGD 更新
    6. 统计 loss 和 accuracy
    """
    model.train()

    # 用来累计整个 epoch 的平均指标
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    start = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        # ==========
        # 从 batch 中取数据并搬到 device
        # ==========
        image = batch["image"].to(device, non_blocking=True)     # [B,1,572,572]
        target = batch["target"].to(device, non_blocking=True)   # [B,388,388]
        weight = batch["weight"].to(device, non_blocking=True)   # [B,388,388]

        # 清空上一轮的梯度
        optimizer.zero_grad(set_to_none=True)

        # ==========
        # 前向传播
        # ==========
        logits = model(image)                                    # [B,2,388,388]

        # ==========
        # 计算论文式 weighted cross entropy
        # ==========
        loss = weighted_pixelwise_cross_entropy(logits, target, weight)

        # ==========
        # 反向传播
        # ==========
        loss.backward()

        # ==========
        # 更新参数
        # ==========
        optimizer.step()

        # ==========
        # 计算当前 batch 的简单准确率
        # ==========
        acc = pixel_accuracy(logits.detach(), target)

        # ==========
        # 统计 epoch 平均值
        # ==========
        running_loss += float(loss.item())
        running_acc += acc
        num_batches += 1
        global_step += 1

        # 按频率打印训练过程
        if batch_idx % print_freq == 0 or batch_idx == 1 or batch_idx == len(loader):
            print(
                f"[Epoch {epoch:03d}] "
                f"[Iter {batch_idx:04d}/{len(loader):04d}] "
                f"loss={loss.item():.6f} "
                f"acc={acc:.4f}"
            )

    elapsed = time.time() - start

    # 返回这个 epoch 的统计结果
    return {
        "epoch_loss": running_loss / max(num_batches, 1),
        "epoch_acc": running_acc / max(num_batches, 1),
        "epoch_time_sec": elapsed,
        "global_step": global_step,
    }


def main() -> None:
    """
    主流程。

    顺序如下：
    1. 读参数
    2. 固定随机种子
    3. 选择设备
    4. 构建 DataLoader
    5. 构建 Model
    6. 构建 Optimizer
    7. 如果需要则恢复训练
    8. 循环训练每个 epoch
    9. 写日志并保存 checkpoint
    """
    args = parse_args()

    # 固定随机性
    set_seed(args.seed)

    # 选择 cpu/cuda/mps
    device = select_device(args.device)

    # 规范化路径
    args.processed_root = args.processed_root.expanduser().resolve()
    args.save_dir = args.save_dir.expanduser().resolve()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # 打印当前训练配置，方便实验记录
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

    # 构建训练 DataLoader
    loader = build_train_loader(args)

    # 构建 U-Net 模型
    model = UNetPaper(
        in_channels=1,   # 灰度图像
        num_classes=2,   # 背景 / 前景
        use_bottleneck_dropout=args.use_bottleneck_dropout,
        bottleneck_dropout_p=args.dropout_p,
    ).to(device)

    # 使用 SGD
    # 这里 momentum=0.99 对应原论文
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # 初始化训练状态
    start_epoch = 1
    global_step = 0
    best_loss = float("inf")

    # 如果传了 resume，则从已有 checkpoint 恢复
    if args.resume:
        ckpt_path = Path(args.resume).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

        last_epoch, global_step, best_loss = load_checkpoint(
            ckpt_path=ckpt_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        start_epoch = last_epoch + 1

        print(f"Resumed from: {ckpt_path}")
        print(f"Start epoch: {start_epoch}, global_step: {global_step}, best_loss: {best_loss:.6f}")

    # 训练日志 csv
    log_csv = args.save_dir / "train_log.csv"

    # 如果文件不存在，就先写表头
    write_header = not log_csv.exists()

    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["epoch", "loss", "acc", "time_sec", "global_step"])

        # ==========
        # epoch 训练主循环
        # ==========
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

            # 更新 global_step
            global_step = int(stats["global_step"])

            # 取出本 epoch 的统计量
            epoch_loss = float(stats["epoch_loss"])
            epoch_acc = float(stats["epoch_acc"])
            epoch_time = float(stats["epoch_time_sec"])

            print(
                f">>> Epoch {epoch:03d} done: "
                f"loss={epoch_loss:.6f}, "
                f"acc={epoch_acc:.4f}, "
                f"time={epoch_time:.2f}s"
            )

            # 写入 csv 日志
            writer.writerow([epoch, epoch_loss, epoch_acc, epoch_time, global_step])
            f.flush()

            # 每个 epoch 都保存 latest
            latest_path = args.save_dir / "latest.pt"
            save_checkpoint(
                save_path=latest_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                args=args,
            )

            # 如果当前 epoch 的 train loss 更好，就更新 best_train_loss.pt
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_path = args.save_dir / "best_train_loss.pt"
                save_checkpoint(
                    save_path=best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_loss=best_loss,
                    args=args,
                )
                print(f"*** New best train loss: {best_loss:.6f}")

            # 按 save_every 频率额外存一个 epoch_xxx.pt
            if epoch % args.save_every == 0:
                epoch_path = args.save_dir / f"epoch_{epoch:03d}.pt"
                save_checkpoint(
                    save_path=epoch_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_loss=best_loss,
                    args=args,
                )

    print("Training finished.")
    print(f"Best train loss: {best_loss:.6f}")
    print(f"Logs saved to: {log_csv}")


if __name__ == "__main__":
    main()
