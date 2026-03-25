from __future__ import annotations

import torch


class DDPMScheduler:
    def __init__(self, timesteps: int, beta_start: float, beta_end: float, device: str):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, batch_size: int, device: str) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return (xt - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)

    def one_step_sample(self, zt: torch.Tensor, pred_noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return (zt - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
