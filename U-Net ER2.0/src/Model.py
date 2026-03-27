#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConvValid(nn.Module):
    """
    U-Net 原文每个 block 的核心单元：
    3x3 valid conv + ReLU
    3x3 valid conv + ReLU

    注意：
    这里 padding=0，也就是 unpadded / valid convolution。
    每经过一个 3x3 conv，特征图长宽各减少 2。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def center_crop(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    对 skip feature 做中心裁剪，使它和上采样后的特征图空间尺寸一致。

    这是 U-Net 原文里的 copy and crop。
    因为编码端一路 valid conv 后，边缘信息会不断损失，
    所以解码端拼接前必须 crop。
    """
    _, _, h, w = x.shape

    if h < target_h or w < target_w:
        raise ValueError(
            f"Cannot crop from {(h, w)} to {(target_h, target_w)}"
        )

    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return x[:, :, top:top + target_h, left:left + target_w]


class UpBlock(nn.Module):
    """
    U-Net 原文解码端一个 stage：
    1. 2x2 up-convolution，把通道数减半，空间尺寸放大 2 倍
    2. crop 对应的 encoder feature
    3. concat
    4. 两个 3x3 valid conv + ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # 原文里的 2x2 up-convolution
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=True,
        )

        # concat 后通道数会变成 out_channels * 2
        self.conv = DoubleConvValid(in_channels=out_channels * 2, out_channels=out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = center_crop(skip, x.shape[-2], x.shape[-1])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNetPaper(nn.Module):
    """
    严格按 U-Net 原论文主干结构实现的 PyTorch 版本。

    通道配置：
    64 -> 128 -> 256 -> 512 -> 1024
    再对称回去。

    默认 num_classes=2：
    背景 / 前景
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        use_bottleneck_dropout: bool = False,
        bottleneck_dropout_p: float = 0.5,
    ):
        super().__init__()

        # Contracting path
        self.enc1 = DoubleConvValid(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConvValid(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConvValid(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConvValid(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConvValid(512, 1024)

        # 论文 3.1 提到 contracting path 末端可加 dropout。
        # 但正文没有把“具体加几层、加在哪里”写死。
        # 这里做成可选开关，默认先关掉。
        self.use_bottleneck_dropout = use_bottleneck_dropout
        self.dropout = nn.Dropout2d(p=bottleneck_dropout_p)

        # Expansive path
        self.up4 = UpBlock(1024, 512)
        self.up3 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up1 = UpBlock(128, 64)

        # Final 1x1 conv
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1, padding=0, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        对应论文里的 sqrt(2 / N) 初始化思想。
        在 PyTorch 里用 kaiming_normal_ 实现。
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)          # 572 -> 568
        p1 = self.pool1(x1)        # 568 -> 284

        x2 = self.enc2(p1)         # 284 -> 280
        p2 = self.pool2(x2)        # 280 -> 140

        x3 = self.enc3(p2)         # 140 -> 136
        p3 = self.pool3(x3)        # 136 -> 68

        x4 = self.enc4(p3)         # 68 -> 64
        p4 = self.pool4(x4)        # 64 -> 32

        x5 = self.bottleneck(p4)   # 32 -> 28

        if self.use_bottleneck_dropout:
            x5 = self.dropout(x5)

        # Decoder
        x = self.up4(x5, x4)       # 28 -> 56, concat crop(64->56), conv -> 52
        x = self.up3(x, x3)        # 52 -> 104, crop(136->104), conv -> 100
        x = self.up2(x, x2)        # 100 -> 200, crop(280->200), conv -> 196
        x = self.up1(x, x1)        # 196 -> 392, crop(568->392), conv -> 388

        logits = self.out_conv(x)  # 388 -> 388
        return logits


if __name__ == "__main__":
    # 用论文经典输入尺寸做 shape 自检
    model = UNetPaper(in_channels=1, num_classes=2, use_bottleneck_dropout=False)
    x = torch.randn(1, 1, 572, 572)
    y = model(x)

    print("input shape :", tuple(x.shape))
    print("output shape:", tuple(y.shape))

    # 按原论文图 1，572 输入应得到 388 输出
    assert y.shape == (1, 2, 388, 388), f"Unexpected output shape: {tuple(y.shape)}"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total params    :", total_params)
    print("trainable params:", trainable_params)
    print("Model self-check passed.")
