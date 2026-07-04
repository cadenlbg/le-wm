from __future__ import annotations

import torch
from torch import nn


class LatentSubgoalACTPolicy(nn.Module):
    """Predict a short latent subgoal first, then predict an action chunk."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int = 512,
        subgoal_depth: int = 3,
        action_depth: int = 4,
        dropout: float = 0.1,
        num_heads: int = 8,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.hidden_dim = int(hidden_dim)

        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.subgoal_query = nn.Parameter(torch.randn(1, self.hidden_dim) * 0.02)
        self.subgoal_role = nn.Parameter(torch.randn(3, self.hidden_dim) * 0.02)

        subgoal_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * self.hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.subgoal_transformer = nn.TransformerEncoder(subgoal_layer, num_layers=subgoal_depth)
        self.subgoal_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

        self.action_queries = nn.Parameter(torch.randn(self.action_horizon, self.hidden_dim) * 0.02)
        self.action_role = nn.Parameter(torch.randn(3 + self.action_horizon, self.hidden_dim) * 0.02)
        action_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * self.hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_transformer = nn.TransformerEncoder(action_layer, num_layers=action_depth)
        self.action_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def forward(self, z_t: torch.Tensor, z_g: torch.Tensor, z_h_teacher: torch.Tensor | None = None, teacher_force_subgoal: bool = False):
        leading_shape = z_t.shape[:-1]
        z_t = z_t.reshape(-1, self.latent_dim)
        z_g = z_g.reshape(-1, self.latent_dim)

        pred_z_h = self.predict_subgoal(z_t, z_g)
        action_z_h = z_h_teacher.reshape(-1, self.latent_dim) if teacher_force_subgoal and z_h_teacher is not None else pred_z_h
        action = self.predict_action(z_t, z_g, action_z_h)

        pred_z_h = pred_z_h.reshape(*leading_shape, self.latent_dim)
        action = action.reshape(*leading_shape, self.action_horizon, self.action_dim)
        return action, pred_z_h

    def predict_subgoal(self, z_t: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        condition = torch.stack([z_t, z_g], dim=1)
        condition = self.latent_proj(condition)
        query = self.subgoal_query.unsqueeze(0).expand(condition.shape[0], -1, -1)
        tokens = torch.cat([condition, query], dim=1)
        tokens = tokens + self.subgoal_role.unsqueeze(0)
        tokens = self.subgoal_transformer(tokens)
        return self.subgoal_head(tokens[:, -1])

    def predict_action(self, z_t: torch.Tensor, z_g: torch.Tensor, z_h: torch.Tensor) -> torch.Tensor:
        condition = torch.stack([z_t, z_g, z_h], dim=1)
        condition = self.latent_proj(condition)
        queries = self.action_queries.unsqueeze(0).expand(condition.shape[0], -1, -1)
        tokens = torch.cat([condition, queries], dim=1)
        tokens = tokens + self.action_role.unsqueeze(0)
        tokens = self.action_transformer(tokens)
        return self.action_head(tokens[:, 3:])

