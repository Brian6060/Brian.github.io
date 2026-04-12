from __future__ import annotations

"""SwinUNet 模型定义

设计目标：
1. 与 ViTUNet 保持统一接口：
   - backbone
   - pretrained
   - num_classes
2. 使用 timm 官方 Swin backbone
3. 支持官方匹配预训练
4. 兼容 timm 不同版本返回的 NHWC / NCHW 特征格式
"""

from typing import List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBNReLU(nn.Module):
    """基础卷积块：Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """上采样 + skip 拼接 + 卷积融合"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_ch + skip_ch, out_ch)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SwinUNet(nn.Module):
    """Swin Transformer + U-Net style decoder"""

    def __init__(
        self,
        backbone: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = False,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone
        self.pretrained = pretrained
        self.num_classes = num_classes

        # 使用 timm 的 features_only 接口直接取多尺度特征
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        feature_info = self.encoder.feature_info
        channels: List[int] = feature_info.channels()
        if len(channels) != 4:
            raise ValueError(f"Expected 4 Swin stages, got {len(channels)}.")

        self.stage_channels = channels
        c1, c2, c3, c4 = channels

        # center + decoder
        self.center = ConvBNReLU(c4, 512)
        self.up1 = UpBlock(512, c3, 256)   # 7 -> 14
        self.up2 = UpBlock(256, c2, 128)   # 14 -> 28
        self.up3 = UpBlock(128, c1, 64)    # 28 -> 56

        self.refine = ConvBNReLU(64, 32)
        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)

    @staticmethod
    def _ensure_nchw(x: Tensor, expected_channels: int) -> Tensor:
        """兼容 timm 可能返回的两种格式：
        1. NCHW: [B, C, H, W]
        2. NHWC: [B, H, W, C]
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D feature map, got shape={tuple(x.shape)}")

        # 已经是 NCHW
        if x.shape[1] == expected_channels:
            return x.contiguous()

        # NHWC -> NCHW
        if x.shape[-1] == expected_channels:
            return x.permute(0, 3, 1, 2).contiguous()

        raise ValueError(
            f"Cannot infer feature layout for shape={tuple(x.shape)}, "
            f"expected_channels={expected_channels}"
        )

    def forward(self, x: Tensor) -> Tensor:
        """输入:
            x: [B, 3, 224, 224]
        输出:
            logits: [B, num_classes, 224, 224]
        """
        feats = self.encoder(x)
        if len(feats) != 4:
            raise RuntimeError(f"Expected 4 feature maps, got {len(feats)}.")

        c1, c2, c3, c4 = self.stage_channels

        f1 = self._ensure_nchw(feats[0], c1)   # 通常 56x56
        f2 = self._ensure_nchw(feats[1], c2)   # 通常 28x28
        f3 = self._ensure_nchw(feats[2], c3)   # 通常 14x14
        f4 = self._ensure_nchw(feats[3], c4)   # 通常 7x7

        x = self.center(f4)   # [B, 512, 7, 7]
        x = self.up1(x, f3)   # [B, 256, 14, 14]
        x = self.up2(x, f2)   # [B, 128, 28, 28]
        x = self.up3(x, f1)   # [B,  64, 56, 56]

        x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        x = self.refine(x)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        logits = self.seg_head(x)
        return logits


if __name__ == "__main__":
    model = SwinUNet(
        backbone="swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=1,
    )
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Input shape :", dummy.shape)
    print("Output shape:", out.shape)
