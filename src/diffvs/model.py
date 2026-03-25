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


class MarkerConditionedDenoiser(nn.Module):
    """A compact U-Net-style denoiser with marker embedding conditioning.

    - Input: H&E image + noisy target marker map
    - Conditioning: marker index embedding broadcast in spatial dims
    - Output: predicted noise for that marker
    """

    def __init__(self, marker_count: int, base_channels: int = 64):
        super().__init__()
        self.marker_embed = nn.Embedding(marker_count, 16)

        self.enc1 = ConvBlock(1 + 1 + 16, base_channels)
        self.down = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.out = nn.Conv2d(base_channels, 1, 1)

    def forward(
        self,
        he: torch.Tensor,
        noisy_marker: torch.Tensor,
        marker_idx: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = noisy_marker.shape
        emb = self.marker_embed(marker_idx).view(b, -1, 1, 1).expand(b, -1, h, w)

        x = torch.cat([he, noisy_marker, emb], dim=1)

        s1 = self.enc1(x)
        s2 = self.enc2(self.down(s1))

        x = self.up(s2)
        x = torch.cat([x, s1], dim=1)
        x = self.dec(x)
        return self.out(x)
