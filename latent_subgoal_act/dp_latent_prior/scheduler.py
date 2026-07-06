from __future__ import annotations

import math

import torch


def cosine_beta_schedule(num_steps: int, s: float = 0.008, device: str | torch.device = "cpu"):
    steps = int(num_steps) + 1
    x = torch.linspace(0, int(num_steps), steps, device=device)
    alpha_bar = torch.cos(((x / int(num_steps)) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-4, 0.999)


class DDPMScheduler:
    def __init__(self, num_steps: int = 100, beta_schedule: str = "cosine", beta_start: float = 1e-4, beta_end: float = 2e-2, device="cpu"):
        self.num_steps = int(num_steps)
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.num_steps, device=device)
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.num_steps, device=device)
        else:
            raise ValueError(f"Unknown beta_schedule={beta_schedule}")
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.beta_schedule = beta_schedule
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def add_noise(self, action: torch.Tensor, step: torch.Tensor, noise: torch.Tensor):
        alpha_bar = self.alpha_bars[step].view(-1, 1, 1)
        return alpha_bar.sqrt() * action + (1.0 - alpha_bar).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, z_t: torch.Tensor, z_g: torch.Tensor, num_samples: int = 1):
        batch_size = z_t.shape[0]
        num_samples = int(num_samples)
        action = torch.randn(
            batch_size * num_samples,
            model.prediction_horizon,
            model.action_dim,
            device=z_t.device,
        )
        flat_z_t = z_t.unsqueeze(1).expand(batch_size, num_samples, -1).reshape(batch_size * num_samples, -1)
        flat_z_g = z_g.unsqueeze(1).expand(batch_size, num_samples, -1).reshape(batch_size * num_samples, -1)
        for step in reversed(range(self.num_steps)):
            step_tensor = torch.full((action.shape[0],), step, device=action.device, dtype=torch.long)
            pred_noise = model(action, step_tensor, flat_z_t, flat_z_g)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            beta = self.betas[step]
            action = (action - beta / (1.0 - alpha_bar).sqrt() * pred_noise) / alpha.sqrt()
            if step > 0:
                action = action + beta.sqrt() * torch.randn_like(action)
        return action.reshape(batch_size, num_samples, model.prediction_horizon, model.action_dim)

