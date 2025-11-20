"""
UNet++ implementation for coronary angiogram segmentation.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
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
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)


class UNetPlusPlus(nn.Module):
    """
    Implementation of UNet++ (Nested U-Net) with optional deep supervision.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        filters: Sequence[int] = (32, 64, 128, 256, 512),
        deep_supervision: bool = False,
        up_mode: str = "bilinear",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision

        self.conv0_0 = ConvBlock(in_channels, filters[0], dropout=dropout)
        self.conv1_0 = ConvBlock(filters[0], filters[1], dropout=dropout)
        self.conv2_0 = ConvBlock(filters[1], filters[2], dropout=dropout)
        self.conv3_0 = ConvBlock(filters[2], filters[3], dropout=dropout)
        self.conv4_0 = ConvBlock(filters[3], filters[4], dropout=dropout)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0])
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1])
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2])
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3])

        self.conv0_2 = ConvBlock(filters[0] * 2 + filters[1], filters[0])
        self.conv1_2 = ConvBlock(filters[1] * 2 + filters[2], filters[1])
        self.conv2_2 = ConvBlock(filters[2] * 2 + filters[3], filters[2])

        self.conv0_3 = ConvBlock(filters[0] * 3 + filters[1], filters[0])
        self.conv1_3 = ConvBlock(filters[1] * 3 + filters[2], filters[1])

        self.conv0_4 = ConvBlock(filters[0] * 4 + filters[1], filters[0])

        if deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        x0_0 = self.conv0_0(x)  # down path
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._upsample(x1_0, x0_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._upsample(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, self._upsample(x1_1, x0_0)], dim=1)
        )

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._upsample(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(
            torch.cat([x1_0, x1_1, self._upsample(x2_1, x1_0)], dim=1)
        )
        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self._upsample(x1_2, x0_0)], dim=1)
        )

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._upsample(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(
            torch.cat([x2_0, x2_1, self._upsample(x3_1, x2_0)], dim=1)
        )
        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, self._upsample(x2_2, x1_0)], dim=1)
        )
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self._upsample(x1_3, x0_0)], dim=1)
        )

        if self.deep_supervision:
            outputs = (
                self.final1(x0_1),
                self.final2(x0_2),
                self.final3(x0_3),
                self.final4(x0_4),
            )
            return outputs
        return self.final(x0_4)

    @staticmethod
    def _upsample(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            source, size=target.shape[2:], mode="bilinear", align_corners=True
        )


