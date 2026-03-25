from __future__ import annotations

import math

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        x = t[:, None].float() * freqs[None, :]
        emb = torch.cat([x.sin(), x.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class MarkerConditionedUNet(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        marker_count: int,
        base_channels: int,
        marker_embed_dim: int,
        time_embed_dim: int,
    ):
        super().__init__()
        self.marker_emb = nn.Embedding(marker_count, marker_embed_dim)
        self.time_emb = TimeEmbedding(time_embed_dim)

        cond_ch = marker_embed_dim + time_embed_dim
        self.in_proj = nn.Conv2d(latent_channels * 2 + cond_ch, base_channels, 1)

        self.enc1 = ResBlock(base_channels, base_channels)
        self.down1 = nn.AvgPool2d(2)
        self.enc2 = ResBlock(base_channels, base_channels * 2)
        self.down2 = nn.AvgPool2d(2)

        self.mid = ResBlock(base_channels * 2, base_channels * 2)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ResBlock(base_channels * 4, base_channels)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ResBlock(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, latent_channels, 1)

    def forward(
        self,
        z_he: torch.Tensor,
        z_noisy_target: torch.Tensor,
        marker_idx: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = z_he.shape
        m = self.marker_emb(marker_idx).view(b, -1, 1, 1).expand(b, -1, h, w)
        tm = self.time_emb(t).view(b, -1, 1, 1).expand(b, -1, h, w)

        x = torch.cat([z_he, z_noisy_target, m, tm], dim=1)
        x = self.in_proj(x)

        s1 = self.enc1(x)
        s2 = self.enc2(self.down1(s1))
        x = self.mid(self.down2(s2))

        x = self.up2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1(x)
        return self.out(x)
