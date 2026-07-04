from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class LatentAwareACTPolicy(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int = 512,
        depth: int = 4,
        dropout: float = 0.1,
        num_heads: int = 8,
        latent_horizon: int = 1,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if latent_horizon < 1:
            raise ValueError("latent_horizon must be >= 1")

        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.hidden_dim = int(hidden_dim)
        self.latent_horizon = int(latent_horizon)

        self.condition_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.action_queries = nn.Parameter(torch.randn(self.action_horizon, self.hidden_dim) * 0.02)
        self.latent_queries = nn.Parameter(torch.randn(self.latent_horizon, self.hidden_dim) * 0.02)
        self.role_embed = nn.Parameter(torch.randn(3 + self.action_horizon + self.latent_horizon, self.hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * self.hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.action_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.latent_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        z_g: torch.Tensor,
        delta_z: Optional[torch.Tensor] = None,
    ):
        if delta_z is None:
            delta_z = z_g - z_t

        leading_shape = z_t.shape[:-1]
        z_t = z_t.reshape(-1, self.latent_dim)
        z_g = z_g.reshape(-1, self.latent_dim)
        delta_z = delta_z.reshape(-1, self.latent_dim)

        cond = torch.stack([z_t, z_g, delta_z], dim=1)
        cond = self.condition_proj(cond)
        queries = torch.cat([self.action_queries, self.latent_queries], dim=0)
        queries = queries.unsqueeze(0).expand(cond.shape[0], -1, -1)
        tokens = torch.cat([cond, queries], dim=1)
        tokens = tokens + self.role_embed.unsqueeze(0)
        tokens = self.transformer(tokens)

        action_tokens = tokens[:, 3 : 3 + self.action_horizon]
        latent_tokens = tokens[:, 3 + self.action_horizon :]
        action = self.action_head(action_tokens).reshape(*leading_shape, self.action_horizon, self.action_dim)
        latent = self.latent_head(latent_tokens).reshape(*leading_shape, self.latent_horizon, self.latent_dim)
        return action, latent

