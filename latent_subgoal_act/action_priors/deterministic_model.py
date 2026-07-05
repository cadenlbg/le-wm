from __future__ import annotations

import torch
from torch import nn


class GoalConditionedActionPrior(nn.Module):
    """Predict an action-chunk mean from current and goal latents."""

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
        self.action_queries = nn.Parameter(torch.randn(self.action_horizon, self.hidden_dim) * 0.02)
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
        self.action_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def forward(self, z_t: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        leading_shape = z_t.shape[:-1]
        z_t = z_t.reshape(-1, self.latent_dim)
        z_g = z_g.reshape(-1, self.latent_dim)

        condition = torch.stack([z_t, z_g], dim=1)
        condition = self.latent_proj(condition)
        queries = self.action_queries.unsqueeze(0).expand(condition.shape[0], -1, -1)
        tokens = torch.cat([condition, queries], dim=1)
        tokens = tokens + self.role_embed.unsqueeze(0)
        tokens = self.transformer(tokens)
        action = self.action_head(tokens[:, 2:])
        return action.reshape(*leading_shape, self.action_horizon, self.action_dim)

