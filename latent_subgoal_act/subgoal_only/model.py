from __future__ import annotations

import torch
from torch import nn


class GoalConditionedSubgoalPredictor(nn.Module):
    """Predict future latent states from current and goal latents only."""

    def __init__(
        self,
        latent_dim: int,
        subgoal_horizon: int,
        hidden_dim: int = 512,
        depth: int = 4,
        dropout: float = 0.1,
        num_heads: int = 8,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.subgoal_horizon = int(subgoal_horizon)
        self.hidden_dim = int(hidden_dim)

        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.subgoal_queries = nn.Parameter(torch.randn(self.subgoal_horizon, self.hidden_dim) * 0.02)
        self.role_embed = nn.Parameter(torch.randn(2 + self.subgoal_horizon, self.hidden_dim) * 0.02)

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
        self.subgoal_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

    def forward(self, z_t: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        leading_shape = z_t.shape[:-1]
        z_t = z_t.reshape(-1, self.latent_dim)
        z_g = z_g.reshape(-1, self.latent_dim)

        condition = torch.stack([z_t, z_g], dim=1)
        condition = self.latent_proj(condition)
        queries = self.subgoal_queries.unsqueeze(0).expand(condition.shape[0], -1, -1)
        tokens = torch.cat([condition, queries], dim=1)
        tokens = tokens + self.role_embed.unsqueeze(0)
        tokens = self.transformer(tokens)
        pred = self.subgoal_head(tokens[:, 2:])
        return pred.reshape(*leading_shape, self.subgoal_horizon, self.latent_dim)

