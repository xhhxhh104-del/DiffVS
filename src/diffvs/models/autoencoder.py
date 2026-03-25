from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyAutoEncoder(nn.Module):
    """Latent autoencoder to mimic LDM workflow (pixel <-> latent)."""

    def __init__(self, in_channels: int = 1, latent_channels: int = 4, base: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBlock(in_channels, base),
            nn.AvgPool2d(2),
            ConvBlock(base, base * 2),
            nn.AvgPool2d(2),
            nn.Conv2d(base * 2, latent_channels, 1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(latent_channels, base * 2, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base * 2, base),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base, base),
            nn.Conv2d(base, 1, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
