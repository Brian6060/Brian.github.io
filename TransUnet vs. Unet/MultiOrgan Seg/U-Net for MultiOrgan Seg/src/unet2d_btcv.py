import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv2D(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down2D(nn.Module):
    """Downscaling with maxpool followed by double conv."""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv2D(in_channels, out_channels, norm_layer=norm_layer),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up2D(nn.Module):
    """Upscaling, skip concatenation, and double conv."""

    def __init__(self, in_channels, skip_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv2D(
            in_channels=(in_channels // 2) + skip_channels,
            out_channels=out_channels,
            norm_layer=norm_layer,
        )

    @staticmethod
    def _pad_to_match(x, target):
        """Pad x so that spatial size matches target."""
        diff_y = target.size(2) - x.size(2)
        diff_x = target.size(3) - x.size(3)

        if diff_y == 0 and diff_x == 0:
            return x

        pad_left = max(diff_x // 2, 0)
        pad_right = max(diff_x - pad_left, 0)
        pad_top = max(diff_y // 2, 0)
        pad_bottom = max(diff_y - pad_top, 0)

        if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
            x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])

        # If the upsampled feature is larger than skip feature, center crop it.
        if x.size(2) > target.size(2):
            crop_top = (x.size(2) - target.size(2)) // 2
            x = x[:, :, crop_top:crop_top + target.size(2), :]
        if x.size(3) > target.size(3):
            crop_left = (x.size(3) - target.size(3)) // 2
            x = x[:, :, :, crop_left:crop_left + target.size(3)]

        return x

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self._pad_to_match(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv2D(nn.Module):
    """Final 1x1 convolution producing segmentation logits."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    """
    Classical 2D U-Net for BTCV multi-organ segmentation.

    Input:
        x: [B, C, H, W]
    Output:
        logits: [B, num_classes, H, W]
    """

    def __init__(self, in_channels=1, num_classes=16, base_channels=64, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.inc = DoubleConv2D(in_channels, c1, norm_layer=norm_layer)
        self.down1 = Down2D(c1, c2, norm_layer=norm_layer)
        self.down2 = Down2D(c2, c3, norm_layer=norm_layer)
        self.down3 = Down2D(c3, c4, norm_layer=norm_layer)
        self.down4 = Down2D(c4, c5, norm_layer=norm_layer)

        self.up1 = Up2D(c5, c4, c4, norm_layer=norm_layer)
        self.up2 = Up2D(c4, c3, c3, norm_layer=norm_layer)
        self.up3 = Up2D(c3, c2, c2, norm_layer=norm_layer)
        self.up4 = Up2D(c2, c1, c1, norm_layer=norm_layer)
        self.outc = OutConv2D(c1, num_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    num_classes = 16
    model = UNet2D(
        in_channels=1,
        num_classes=num_classes,
        base_channels=64,
    )

    x = torch.randn(2, 1, 320, 320)
    output = model(x)

    print("output.shape:", output.shape)
    print("trainable parameters:", count_parameters(model))
