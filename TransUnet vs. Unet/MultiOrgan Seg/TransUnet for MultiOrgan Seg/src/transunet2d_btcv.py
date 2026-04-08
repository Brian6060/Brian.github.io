#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock2D(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock2D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock2D((in_channels // 2) + skip_channels, out_channels)

    @staticmethod
    def _pad_or_crop(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        diff_h = ref.size(2) - x.size(2)
        diff_w = ref.size(3) - x.size(3)

        if diff_h > 0 or diff_w > 0:
            x = F.pad(
                x,
                [
                    max(diff_w // 2, 0),
                    max(diff_w - diff_w // 2, 0),
                    max(diff_h // 2, 0),
                    max(diff_h - diff_h // 2, 0),
                ],
            )

        if x.size(2) > ref.size(2) or x.size(3) > ref.size(3):
            start_h = max((x.size(2) - ref.size(2)) // 2, 0)
            start_w = max((x.size(3) - ref.size(3)) // 2, 0)
            x = x[:, :, start_h:start_h + ref.size(2), start_w:start_w + ref.size(3)]
        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self._pad_or_crop(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class PatchEmbedding2D(nn.Module):
    """
    Feature-level tokenization. By default uses patch_size=1, so every feature
    location becomes one token without patchifying the raw input image.
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 1) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.proj(x)
        h, w = x.shape[2], x.shape[3]
        tokens = x.flatten(2).transpose(1, 2)
        return tokens, (h, w)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TransUNet2D(nn.Module):
    """
    Encoder-only TransUNet:
    1. CNN encoder
    2. Transformer encoder over high-level CNN feature tokens
    3. U-Net style decoder with skip connections
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 64,
        embed_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_transformer_layers: int = 4,
        patch_size: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.inc = ConvBlock2D(in_channels, c1)
        self.down1 = DownBlock2D(c1, c2)
        self.down2 = DownBlock2D(c2, c3)
        self.down3 = DownBlock2D(c3, c4)
        self.down4 = DownBlock2D(c4, c5)

        self.patch_embed = PatchEmbedding2D(c5, embed_dim, patch_size=patch_size)
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_layers=num_transformer_layers,
            dropout=dropout,
        )
        self.transformer_to_feature = nn.Conv2d(embed_dim, c5, kernel_size=1)

        self.up1 = UpBlock2D(c5, c4, c4)
        self.up2 = UpBlock2D(c4, c3, c3)
        self.up3 = UpBlock2D(c3, c2, c2)
        self.up4 = UpBlock2D(c2, c1, c1)
        self.outc = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        tokens, (h, w) = self.patch_embed(x5)
        tokens = self.transformer(tokens)
        x = tokens.transpose(1, 2).reshape(tokens.size(0), tokens.size(2), h, w)
        x = self.transformer_to_feature(x)

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TransUNet2D(
        in_channels=1,
        num_classes=16,
        base_channels=64,
        embed_dim=512,
        num_heads=8,
        num_transformer_layers=4,
    )
    x = torch.randn(2, 1, 320, 320)
    out = model(x)
    print("output.shape:", out.shape)
    print("parameters:", count_parameters(model))
