from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image

from diffvs.config import ExperimentConfig
from diffvs.diffusion.scheduler import DDPMScheduler
from diffvs.models import MarkerConditionedUNet, TinyAutoEncoder


def _read_gray_tif(path: Path, image_size: int) -> np.ndarray:
    arr = tifffile.imread(path).astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., :3].mean(axis=-1)
        else:
            arr = arr.mean(axis=0)
    arr = Image.fromarray(arr).resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(arr, dtype=np.float32)
    vmin, vmax = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    arr = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    return arr


class DiffVSInferencer:
    def __init__(self, cfg: ExperimentConfig, checkpoint: Path):
        self.cfg = cfg
        self.device = cfg.train.device if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(checkpoint, map_location=self.device)
        self.marker_to_idx = ckpt["marker_to_idx"]

        self.ae = TinyAutoEncoder(latent_channels=cfg.model.latent_channels).to(self.device)
        self.unet = MarkerConditionedUNet(
            latent_channels=cfg.model.latent_channels,
            marker_count=len(self.marker_to_idx),
            base_channels=cfg.model.unet_base_channels,
            marker_embed_dim=cfg.model.marker_embed_dim,
            time_embed_dim=cfg.model.time_embed_dim,
        ).to(self.device)

        self.ae.load_state_dict(ckpt["ae"])
        self.unet.load_state_dict(ckpt["unet"])
        self.ae.eval()
        self.unet.eval()

        self.scheduler = DDPMScheduler(
            cfg.diffusion.timesteps,
            cfg.diffusion.beta_start,
            cfg.diffusion.beta_end,
            self.device,
        )

    @torch.no_grad()
    def predict_all_markers(self, he_tif: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        he = _read_gray_tif(he_tif, self.cfg.data.image_size)
        he_t = torch.from_numpy(he)[None, None, ...].to(self.device)
        z_he = self.ae.encode(he_t)

        t = torch.full((1,), self.cfg.diffusion.timesteps - 1, dtype=torch.long, device=self.device)

        for marker_name, marker_idx in self.marker_to_idx.items():
            zt = torch.randn((1, self.cfg.model.latent_channels, z_he.shape[-2], z_he.shape[-1]), device=self.device)
            idx = torch.tensor([marker_idx], dtype=torch.long, device=self.device)

            pred_noise = self.unet(z_he, zt, idx, t)
            z0 = self.scheduler.one_step_sample(zt, pred_noise, t)
            pred = self.ae.decode(z0).squeeze().cpu().numpy()

            out_npy = output_dir / f"pred_{marker_name}.npy"
            out_tif = output_dir / f"pred_{marker_name}.tif"
            np.save(out_npy, pred)
            tifffile.imwrite(out_tif, (np.clip(pred, 0, 1) * 65535).astype(np.uint16))

        (output_dir / "markers.json").write_text(json.dumps(self.marker_to_idx, indent=2))
