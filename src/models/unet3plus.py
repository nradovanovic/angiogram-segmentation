"""
UNet 3+ implementation for coronary angiogram segmentation.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)


def _resize(
    tensor: torch.Tensor, target_shape: torch.Size, mode: str = "bilinear"
) -> torch.Tensor:
    if tensor.shape[-2:] == target_shape[-2:]:
        return tensor
    return F.interpolate(tensor, size=target_shape[-2:], mode=mode, align_corners=True)


class UNet3Plus(nn.Module):
    """
    UNet 3+ implementation with deep supervision support.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        filters: Sequence[int] = (32, 64, 128, 256, 512),
        cat_channels: int = 32,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        if len(filters) != 5:
            raise ValueError("UNet3Plus expects five filter values for the encoder.")
        self.deep_supervision = deep_supervision
        self.cat_channels = cat_channels

        self.encoder1 = ConvBlock(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ConvBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = ConvBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = ConvBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)
        self.encoder5 = ConvBlock(filters[3], filters[4])

        self.h1_d4 = self._make_stage(filters[0], filters[0])
        self.h2_d4 = self._make_stage(filters[1], filters[0])
        self.h3_d4 = self._make_stage(filters[2], filters[0])
        self.h4_d4 = self._make_stage(filters[3], filters[0])
        self.h5_d4 = self._make_stage(filters[4], filters[0])

        self.h1_d3 = self._make_stage(filters[0], filters[1])
        self.h2_d3 = self._make_stage(filters[1], filters[1])
        self.h3_d3 = self._make_stage(filters[2], filters[1])
        self.h4_d3 = self._make_stage(filters[3], filters[1])
        self.h5_d3 = self._make_stage(filters[4], filters[1])

        self.h1_d2 = self._make_stage(filters[0], filters[2])
        self.h2_d2 = self._make_stage(filters[1], filters[2])
        self.h3_d2 = self._make_stage(filters[2], filters[2])
        self.h4_d2 = self._make_stage(filters[3], filters[2])
        self.h5_d2 = self._make_stage(filters[4], filters[2])

        self.h1_d1 = self._make_stage(filters[0], filters[3])
        self.h2_d1 = self._make_stage(filters[1], filters[3])
        self.h3_d1 = self._make_stage(filters[2], filters[3])
        self.h4_d1 = self._make_stage(filters[3], filters[3])
        self.h5_d1 = self._make_stage(filters[4], filters[3])

        self.h1_d0 = self._make_stage(filters[0], filters[4])
        self.h2_d0 = self._make_stage(filters[1], filters[4])
        self.h3_d0 = self._make_stage(filters[2], filters[4])
        self.h4_d0 = self._make_stage(filters[3], filters[4])
        self.h5_d0 = self._make_stage(filters[4], filters[4])

        concat_channels_1 = cat_channels * 5
        self.conv_d4 = ConvBlock(concat_channels_1, filters[3])
        self.conv_d3 = ConvBlock(concat_channels_1, filters[2])
        self.conv_d2 = ConvBlock(concat_channels_1, filters[1])
        self.conv_d1 = ConvBlock(concat_channels_1, filters[0])

        self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

        if deep_supervision:
            self.ds_out1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.ds_out2 = nn.Conv2d(filters[1], out_channels, kernel_size=1)
            self.ds_out3 = nn.Conv2d(filters[2], out_channels, kernel_size=1)
            self.ds_out4 = nn.Conv2d(filters[3], out_channels, kernel_size=1)

    def _make_stage(self, in_channels: int, target_filters: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, self.cat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.cat_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        h1 = self.encoder1(x)
        h2 = self.encoder2(self.pool1(h1))
        h3 = self.encoder3(self.pool2(h2))
        h4 = self.encoder4(self.pool3(h3))
        h5 = self.encoder5(self.pool4(h4))

        # Stage 4
        size4 = h4.size()
        d4_1 = torch.cat(
            [
                self.h1_d4(_resize(h1, size4)),
                self.h2_d4(_resize(h2, size4)),
                self.h3_d4(_resize(h3, size4)),
                self.h4_d4(h4),
                self.h5_d4(F.interpolate(h5, size=size4[-2:], mode="bilinear", align_corners=True)),
            ],
            dim=1,
        )
        d4 = self.conv_d4(d4_1)

        # Stage 3
        size3 = h3.size()
        d3_1 = torch.cat(
            [
                self.h1_d3(_resize(h1, size3)),
                self.h2_d3(_resize(h2, size3)),
                self.h3_d3(h3),
                self.h4_d3(_resize(h4, size3)),
                self.h5_d3(_resize(h5, size3)),
            ],
            dim=1,
        )
        d3 = self.conv_d3(d3_1)

        # Stage 2
        size2 = h2.size()
        d2_1 = torch.cat(
            [
                self.h1_d2(_resize(h1, size2)),
                self.h2_d2(h2),
                self.h3_d2(_resize(h3, size2)),
                self.h4_d2(_resize(h4, size2)),
                self.h5_d2(_resize(h5, size2)),
            ],
            dim=1,
        )
        d2 = self.conv_d2(d2_1)

        # Stage 1
        size1 = h1.size()
        d1_1 = torch.cat(
            [
                self.h1_d1(h1),
                self.h2_d1(_resize(h2, size1)),
                self.h3_d1(_resize(h3, size1)),
                self.h4_d1(_resize(h4, size1)),
                self.h5_d1(_resize(h5, size1)),
            ],
            dim=1,
        )
        d1 = self.conv_d1(d1_1)

        if self.deep_supervision:
            ds1 = self.ds_out1(d1)
            ds2 = self.ds_out2(d2)
            ds3 = self.ds_out3(d3)
            ds4 = self.ds_out4(d4)
            return ds1, ds2, ds3, ds4
        return self.final(d1)


