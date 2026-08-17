"""Exact canopy-segmentation architecture used for the archived atlas.

The network returns probabilities because the recovered training and inference
implementation applied a sigmoid in the model forward pass. This behaviour is
retained for checkpoint compatibility.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class MorphoAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 4) -> None:
        super().__init__()
        reduced = in_channels // reduction_ratio
        self.channel_pool = nn.Sequential(
            nn.Conv2d(in_channels, reduced, 1),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
        )
        self.dilation_conv = nn.Sequential(
            nn.Conv2d(reduced, 1, 3, padding=2, dilation=2),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.erosion_conv = nn.Sequential(
            nn.Conv2d(reduced, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.fusion = nn.Conv2d(2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compressed = self.channel_pool(x)
        dilation = self.dilation_conv(compressed)
        erosion = self.erosion_conv(compressed)
        boundary = self.edge_conv(dilation - erosion)
        attention = torch.sigmoid(self.fusion(torch.cat([boundary, dilation], dim=1)))
        return x * attention


class DirectionCentralityModule(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        reduced = in_channels // 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, reduced, 3, padding=1),
            nn.GroupNorm(4, reduced),
            nn.ReLU(),
        )
        self.direction_conv = nn.Conv2d(reduced, 8, kernel_size=3, padding=1, groups=8)
        nn.init.constant_(self.direction_conv.weight, 0)
        for index in range(8):
            cosine = np.cos(index * np.pi / 4)
            self.direction_conv.weight.data[index, 0] = torch.tensor(
                [[cosine, 0, -cosine], [0, 0, 0], [-cosine, 0, cosine]]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        responses = self.direction_conv(self.conv(x))
        mean = responses.mean(dim=1, keepdim=True)
        variance = torch.mean((responses - mean) ** 2, dim=1, keepdim=True)
        return torch.sigmoid(variance)


class UNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = True) -> None:
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.dcm1 = DirectionCentralityModule(64)
        self.dcm2 = DirectionCentralityModule(128)
        self.dcm3 = DirectionCentralityModule(256)
        self.MorphoAttention1 = MorphoAttention(64)
        self.MorphoAttention2 = MorphoAttention(128)
        self.MorphoAttention3 = MorphoAttention(256)
        self.MorphoAttention4 = MorphoAttention(512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw1 = self.inc(x)
        x1 = self.MorphoAttention1(raw1) * self.dcm1(raw1)
        raw2 = self.down1(x1)
        x2 = self.MorphoAttention2(raw2) * self.dcm2(raw2)
        raw3 = self.down2(x2)
        x3 = self.MorphoAttention3(raw3) * self.dcm3(raw3)
        raw4 = self.down3(x3)
        x4 = self.MorphoAttention4(raw4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.sigmoid(self.outc(x))


def build_model(n_channels: int = 3, n_classes: int = 1, bilinear: bool = True) -> UNet:
    return UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
