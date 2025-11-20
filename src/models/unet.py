"""
Baseline U-Net implementation for coronary angiogram segmentation.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels: int, out_channels: int, dropout: float = 0.0) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.insert(3, nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    """
    Standard U-Net encoder-decoder with skip connections.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        filters: Sequence[int] = (64, 128, 256, 512, 1024),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(filters) != 5:
            raise ValueError("UNet expects five filter values.")

        self.enc1 = conv_block(in_channels, filters[0], dropout=dropout)
        self.enc2 = conv_block(filters[0], filters[1], dropout=dropout)
        self.enc3 = conv_block(filters[1], filters[2], dropout=dropout)
        self.enc4 = conv_block(filters[2], filters[3], dropout=dropout)
        self.bottleneck = conv_block(filters[3], filters[4], dropout=dropout)

        self.pool = nn.MaxPool2d(2, 2)

        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.dec4 = conv_block(filters[4], filters[3], dropout=dropout)

        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.dec3 = conv_block(filters[3], filters[2], dropout=dropout)

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.dec2 = conv_block(filters[2], filters[1], dropout=dropout)

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.dec1 = conv_block(filters[1], filters[0], dropout=dropout)

        self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, self._crop(e4, d4)], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, self._crop(e3, d3)], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, self._crop(e2, d2)], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, self._crop(e1, d1)], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)

    @staticmethod
    def _crop(enc_feature: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if enc_feature.shape[2:] == target.shape[2:]:
            return enc_feature
        _, _, h, w = target.shape
        return F.center_crop(enc_feature, (h, w))


__all__ = ["UNet"]


