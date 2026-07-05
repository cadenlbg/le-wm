from __future__ import annotations

import math

import torch
from torch import nn


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, emb.new_zeros(emb.shape[0], 1)], dim=1)
    return emb


class GoalConditionedDiffusionPrior(nn.Module):
    """Denoise action chunks conditioned only on z_t and z_g."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int = 512,
        depth: int = 4,
        dropout: float = 0.1,
        num_heads: int = 8,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.hidden_dim = int(hidden_dim)

        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.action_proj = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.role_embed = nn.Parameter(torch.randn(2 + self.action_horizon, self.hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * self.hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.noise_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def forward(self, noisy_action: torch.Tensor, diffusion_step: torch.Tensor, z_t: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        leading_shape = noisy_action.shape[:-2]
        noisy_action = noisy_action.reshape(-1, self.action_horizon, self.action_dim)
        z_t = z_t.reshape(-1, self.latent_dim)
        z_g = z_g.reshape(-1, self.latent_dim)
        diffusion_step = diffusion_step.reshape(-1)

        condition = torch.stack([z_t, z_g], dim=1)
        condition = self.latent_proj(condition)
        action_tokens = self.action_proj(noisy_action)
        time = self.time_mlp(timestep_embedding(diffusion_step, self.hidden_dim)).unsqueeze(1)
        tokens = torch.cat([condition, action_tokens], dim=1)
        tokens = tokens + self.role_embed.unsqueeze(0) + time
        tokens = self.transformer(tokens)
        pred_noise = self.noise_head(tokens[:, 2:])
        return pred_noise.reshape(*leading_shape, self.action_horizon, self.action_dim)


class DiffusionSchedule:
    def __init__(self, num_steps: int = 50, beta_start: float = 1e-4, beta_end: float = 2e-2, device: str | torch.device = "cpu"):
        self.num_steps = int(num_steps)
        betas = torch.linspace(beta_start, beta_end, self.num_steps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
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
        device = z_t.device
        action = torch.randn(
            batch_size * int(num_samples),
            model.action_horizon,
            model.action_dim,
            device=device,
        )
        flat_z_t = z_t.unsqueeze(1).expand(batch_size, int(num_samples), -1).reshape(batch_size * int(num_samples), -1)
        flat_z_g = z_g.unsqueeze(1).expand(batch_size, int(num_samples), -1).reshape(batch_size * int(num_samples), -1)
        for step in reversed(range(self.num_steps)):
            step_tensor = torch.full((action.shape[0],), step, device=device, dtype=torch.long)
            pred_noise = model(action, step_tensor, flat_z_t, flat_z_g)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            beta = self.betas[step]
            action = (action - beta / (1.0 - alpha_bar).sqrt() * pred_noise) / alpha.sqrt()
            if step > 0:
                action = action + beta.sqrt() * torch.randn_like(action)
        return action.reshape(batch_size, int(num_samples), model.action_horizon, model.action_dim)
