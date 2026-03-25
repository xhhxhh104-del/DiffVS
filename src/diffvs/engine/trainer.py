from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffvs.config import ExperimentConfig
from diffvs.data import HEMITMarkerwiseDataset, discover_hemit_samples
from diffvs.diffusion.scheduler import DDPMScheduler
from diffvs.models import MarkerConditionedUNet, TinyAutoEncoder


class DiffVSTrainer:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = cfg.train.device if torch.cuda.is_available() else "cpu"

        samples, markers = discover_hemit_samples(
            cfg.data.root,
            cfg.data.he_glob,
            cfg.data.mihc_glob,
            cfg.data.marker_from_name_sep,
            cfg.data.marker_from_name_index,
        )
        split = int(len(samples) * cfg.data.train_split)
        train_samples = samples[:split]
        val_samples = samples[split:] if split < len(samples) else samples[: min(16, len(samples))]

        self.marker_to_idx = {m: i for i, m in enumerate(markers)}

        self.train_ds = HEMITMarkerwiseDataset(train_samples, self.marker_to_idx, cfg.data.image_size)
        self.val_ds = HEMITMarkerwiseDataset(val_samples, self.marker_to_idx, cfg.data.image_size)

        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
            drop_last=True,
        )
        self.val_dl = DataLoader(
            self.val_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            drop_last=False,
        )

        self.ae = TinyAutoEncoder(latent_channels=cfg.model.latent_channels).to(self.device)
        self.unet = MarkerConditionedUNet(
            latent_channels=cfg.model.latent_channels,
            marker_count=len(self.marker_to_idx),
            base_channels=cfg.model.unet_base_channels,
            marker_embed_dim=cfg.model.marker_embed_dim,
            time_embed_dim=cfg.model.time_embed_dim,
        ).to(self.device)

        self.scheduler = DDPMScheduler(
            cfg.diffusion.timesteps,
            cfg.diffusion.beta_start,
            cfg.diffusion.beta_end,
            self.device,
        )

        self.optim = AdamW(
            list(self.ae.parameters()) + list(self.unet.parameters()),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

        self.output_dir = cfg.train.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        meta = {
            "marker_to_idx": self.marker_to_idx,
            "config": json.loads(json.dumps(asdict(cfg), default=str)),
        }
        (self.output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    def _to_device(self, batch):
        return (
            batch["he"].to(self.device),
            batch["target"].to(self.device),
            batch["marker_idx"].to(self.device),
        )

    def _loss_diffusion_stage(self, he, target, marker_idx):
        z_he = self.ae.encode(he)
        z_tgt = self.ae.encode(target)

        t = self.scheduler.sample_timesteps(he.shape[0], self.device)
        noise = torch.randn_like(z_tgt)
        z_noisy = self.scheduler.q_sample(z_tgt, t, noise)

        pred_noise = self.unet(z_he, z_noisy, marker_idx, t)
        pred_x0 = self.scheduler.predict_x0(z_noisy, t, pred_noise)

        loss_noise = F.mse_loss(pred_noise, noise)
        loss_x0 = F.l1_loss(pred_x0, z_tgt)
        total = self.cfg.loss.lambda_noise * loss_noise + self.cfg.loss.lambda_x0 * loss_x0
        return total, {"loss_noise": loss_noise.item(), "loss_x0": loss_x0.item()}

    def _loss_onestep_stage(self, he, target, marker_idx):
        z_he = self.ae.encode(he)
        z_tgt = self.ae.encode(target)

        t = torch.full((he.shape[0],), self.cfg.diffusion.timesteps - 1, device=self.device, dtype=torch.long)
        zt = torch.randn_like(z_tgt)

        pred_noise = self.unet(z_he, zt, marker_idx, t)
        pred_z0 = self.scheduler.one_step_sample(zt, pred_noise, t)
        pred_img = self.ae.decode(pred_z0)

        loss_l1 = F.l1_loss(pred_img, target)
        loss_mse = F.mse_loss(pred_img, target)
        total = self.cfg.loss.lambda_l1 * loss_l1 + self.cfg.loss.lambda_mse * loss_mse
        return total, {"loss_l1": loss_l1.item(), "loss_mse": loss_mse.item()}

    def _run_epoch(self, stage: str, epoch: int):
        self.ae.train()
        self.unet.train()
        pbar = tqdm(self.train_dl, desc=f"{stage} epoch {epoch:03d}")
        logs = []

        for batch in pbar:
            he, target, marker_idx = self._to_device(batch)

            if stage == "diffusion":
                loss, detail = self._loss_diffusion_stage(he, target, marker_idx)
            else:
                loss, detail = self._loss_onestep_stage(he, target, marker_idx)

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()

            logs.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = sum(logs) / max(len(logs), 1)
        print(f"[train/{stage}] epoch={epoch} avg_loss={avg:.6f}")

    @torch.no_grad()
    def _validate(self, stage: str, epoch: int):
        self.ae.eval()
        self.unet.eval()
        losses = []

        for batch in self.val_dl:
            he, target, marker_idx = self._to_device(batch)
            if stage == "diffusion":
                loss, _ = self._loss_diffusion_stage(he, target, marker_idx)
            else:
                loss, _ = self._loss_onestep_stage(he, target, marker_idx)
            losses.append(float(loss.item()))

        avg = sum(losses) / max(len(losses), 1)
        print(f"[val/{stage}] epoch={epoch} avg_loss={avg:.6f}")

    def _save_checkpoint(self, stage: str, epoch: int):
        ckpt = {
            "ae": self.ae.state_dict(),
            "unet": self.unet.state_dict(),
            "marker_to_idx": self.marker_to_idx,
            "config": json.loads(json.dumps(asdict(self.cfg), default=str)),
            "stage": stage,
            "epoch": epoch,
        }
        path = self.output_dir / "checkpoints" / f"{stage}_epoch_{epoch:03d}.pt"
        torch.save(ckpt, path)

    def fit(self):
        for ep in range(1, self.cfg.train.epochs_diffusion + 1):
            self._run_epoch("diffusion", ep)
            self._validate("diffusion", ep)
            if ep % self.cfg.train.save_every == 0:
                self._save_checkpoint("diffusion", ep)

        for ep in range(1, self.cfg.train.epochs_onestep + 1):
            self._run_epoch("onestep", ep)
            self._validate("onestep", ep)
            if ep % self.cfg.train.save_every == 0:
                self._save_checkpoint("onestep", ep)
