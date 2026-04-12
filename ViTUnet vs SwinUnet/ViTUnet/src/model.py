from __future__ import annotations

"""ViTUNet 模型定义

设计目标：
1. 兼容 timm 官方 ViT backbone
2. 支持 pretrained=True 自动加载官方匹配预训练
3. 对外接口与 train/test/inference 保持一致：
   - backbone
   - pretrained
   - num_classes
4. 使用 U-Net 风格 decoder 做分割预测
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


class ViTUNet(nn.Module):
    """ViT + U-Net style decoder

    说明：
    - encoder 使用 timm ViT
    - 从多个 transformer block 抽取中间 token 特征
    - token -> feature map
    - 通过 U-Net 风格 decoder 恢复分割图
    """

    def __init__(
        self,
        backbone: str = "vit_small_patch16_224",
        pretrained: bool = False,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone
        self.pretrained = pretrained
        self.num_classes = num_classes

        # 使用 timm 创建 ViT backbone
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
        )

        if not hasattr(self.encoder, "patch_embed"):
            raise ValueError(f"Backbone {backbone} does not expose patch_embed.")
        if not hasattr(self.encoder, "blocks"):
            raise ValueError(f"Backbone {backbone} does not expose transformer blocks.")

        self.embed_dim = self.encoder.embed_dim

        # 当前实现默认基于标准 12-block ViT
        num_blocks = len(self.encoder.blocks)
        if num_blocks < 12:
            raise ValueError(
                f"Current implementation expects >=12 transformer blocks, got {num_blocks}."
            )

        # 抽取 4 个 block 输出作为 skip 特征
        self.hook_block_ids = [2, 5, 8, 11]

        # 将 token map 投影到不同通道数，便于 decoder 融合
        self.skip_proj1 = nn.Conv2d(self.embed_dim, 512, kernel_size=1)
        self.skip_proj2 = nn.Conv2d(self.embed_dim, 256, kernel_size=1)
        self.skip_proj3 = nn.Conv2d(self.embed_dim, 128, kernel_size=1)
        self.skip_proj4 = nn.Conv2d(self.embed_dim, 64, kernel_size=1)

        # center 特征
        self.center = ConvBNReLU(self.embed_dim, 512)

        # decoder
        self.up1 = UpBlock(512, 512, 256)
        self.up2 = UpBlock(256, 256, 128)
        self.up3 = UpBlock(128, 128, 64)
        self.up4 = UpBlock(64, 64, 32)

        # 最终分割头
        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def _tokens_to_map(self, x: Tensor) -> Tensor:
        """将 token 序列 [B, N, C] 转成 2D 特征图 [B, C, H, W]"""
        if x.ndim != 3:
            raise ValueError(f"Expected token tensor [B,N,C], got shape={tuple(x.shape)}")

        # 如果带 cls token，则去掉
        if x.shape[1] % 2 != 0:
            x = x[:, 1:, :]

        b, n, c = x.shape
        hw = int(n ** 0.5)
        if hw * hw != n:
            raise ValueError(f"Token count {n} is not a square number after cls removal.")

        x = x.transpose(1, 2).contiguous().view(b, c, hw, hw)
        return x

    def _forward_encoder(self, x: Tensor) -> List[Tensor]:
        """前向经过 ViT encoder，并抽取多个 block 的中间输出"""
        x = self.encoder.patch_embed(x)  # [B, N, C]

        if hasattr(self.encoder, "cls_token") and self.encoder.cls_token is not None:
            cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if hasattr(self.encoder, "pos_embed") and self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed

        if hasattr(self.encoder, "pos_drop"):
            x = self.encoder.pos_drop(x)

        features: List[Tensor] = []
        for i, blk in enumerate(self.encoder.blocks):
            x = blk(x)
            if i in self.hook_block_ids:
                features.append(x)

        if hasattr(self.encoder, "norm"):
            x = self.encoder.norm(x)

        # 用最终 norm 后特征替换最后一级特征，数值更稳
        features[-1] = x
        return features

    def forward(self, x: Tensor) -> Tensor:
        """
        输入:
            x: [B, 3, 224, 224]
        输出:
            logits: [B, num_classes, 224, 224]
        """
        features = self._forward_encoder(x)
        if len(features) != 4:
            raise RuntimeError(f"Expected 4 skip features, got {len(features)}.")

        # token -> 2D feature map，初始分辨率通常是 14x14
        f1 = self._tokens_to_map(features[0])
        f2 = self._tokens_to_map(features[1])
        f3 = self._tokens_to_map(features[2])
        f4 = self._tokens_to_map(features[3])

        # 投影成 decoder 所需的 skip 通道数
        s1 = self.skip_proj1(f1)  # [B, 512, 14, 14]
        s2 = self.skip_proj2(f2)  # [B, 256, 14, 14]
        s3 = self.skip_proj3(f3)  # [B, 128, 14, 14]
        s4 = self.skip_proj4(f4)  # [B,  64, 14, 14]

        # 为了形成分层 decoder，这里用上采样构造多尺度 skip
        s1_up = s1
        s2_up = F.interpolate(s2, size=(28, 28), mode="bilinear", align_corners=False)
        s3_up = F.interpolate(s3, size=(56, 56), mode="bilinear", align_corners=False)
        s4_up = F.interpolate(s4, size=(112, 112), mode="bilinear", align_corners=False)

        center = self.center(f4)     # [B, 512, 14, 14]
        x = self.up1(center, s1_up)  # -> [B, 256, 14, 14]
        x = self.up2(x, s2_up)       # -> [B, 128, 28, 28]
        x = self.up3(x, s3_up)       # -> [B,  64, 56, 56]
        x = self.up4(x, s4_up)       # -> [B,  32,112,112]

        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        logits = self.seg_head(x)
        return logits


if __name__ == "__main__":
    model = ViTUNet(
        backbone="vit_small_patch16_224",
        pretrained=True,
        num_classes=1,
    )
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Input shape :", dummy.shape)
    print("Output shape:", out.shape)
