"""
Simplified TransUNet implementation combining CNN encoder and Transformer bottleneck.
"""

from __future__ import annotations

from typing import Sequence, Tuple

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


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        return x


class TransformerBottleneck(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.encoder(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TransUNet(nn.Module):
    """
    Transformer-based U-Net variant for angiogram segmentation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: Tuple[int, int] = (512, 512),
        filters: Sequence[int] = (32, 64, 128, 256),
        embed_dim: int = 256,
        transformer_depth: int = 4,
        num_heads: int = 8,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        height, width = img_size
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size.")

        self.img_size = img_size
        self.enc1 = ConvBlock(in_channels, filters[0])
        self.enc2 = ConvBlock(filters[0], filters[1])
        self.enc3 = ConvBlock(filters[1], filters[2])
        self.enc4 = ConvBlock(filters[2], filters[3])

        self.pool = nn.MaxPool2d(2)

        downscale = 2 ** (len(filters) - 1)
        bottleneck_hw = (height // downscale, width // downscale)
        patch_kernel = max(1, patch_size // downscale)
        if bottleneck_hw[0] % patch_kernel != 0 or bottleneck_hw[1] % patch_kernel != 0:
            raise ValueError(
                "Incompatible patch/kernel configuration. "
                "Ensure patch_size leads to integer number of tokens."
            )
        token_hw = (bottleneck_hw[0] // patch_kernel, bottleneck_hw[1] // patch_kernel)
        self.patch_embed = PatchEmbedding(filters[3], embed_dim, patch_size=patch_kernel)

        num_patches = token_hw[0] * token_hw[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.transformer = TransformerBottleneck(
            embed_dim=embed_dim,
            depth=transformer_depth,
            num_heads=num_heads,
        )
        self.proj_back = nn.Conv2d(embed_dim, filters[3], kernel_size=1)
        self._bottleneck_hw = bottleneck_hw
        self._token_hw = token_hw

        self.up1 = UpBlock(filters[3], filters[2], filters[2])
        self.up2 = UpBlock(filters[2], filters[1], filters[1])
        self.up3 = UpBlock(filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.enc1(x)
        h2 = self.enc2(self.pool(h1))
        h3 = self.enc3(self.pool(h2))
        h4 = self.enc4(self.pool(h3))

        tokens = self.patch_embed(h4)  # B, N, C
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]
        tokens = self.transformer(tokens)

        b, _, c = tokens.shape
        tokens = tokens.transpose(1, 2).contiguous()
        tokens = tokens.view(b, c, self._token_hw[0], self._token_hw[1])
        tokens = F.interpolate(
            tokens,
            size=self._bottleneck_hw,
            mode="bilinear",
            align_corners=True,
        )
        bottleneck = self.proj_back(tokens)

        d3 = self.up1(bottleneck, h3)
        d2 = self.up2(d3, h2)
        d1 = self.up3(d2, h1)
        return self.final(d1)


