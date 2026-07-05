from __future__ import annotations

import torch
from torch import nn

from latent_subgoal_act.model import LatentSubgoalACTPolicy


class LatentSubgoalACTPriorPolicy(LatentSubgoalACTPolicy):
    """Variant that predicts a Gaussian action prior in normalized action space.

    This class is intentionally separate from `LatentSubgoalACTPolicy` so the
    current main experiments remain unchanged.
    """

    def __init__(
        self,
        *args,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        init_log_std: float = -0.7,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.action_log_std_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        nn.init.constant_(self.action_log_std_head[-1].bias, float(init_log_std))

    def predict_action_distribution(self, z_t: torch.Tensor, z_g: torch.Tensor, z_h_seq: torch.Tensor):
        condition = torch.cat([torch.stack([z_t, z_g], dim=1), z_h_seq], dim=1)
        condition = self.latent_proj(condition)
        queries = self.action_queries.unsqueeze(0).expand(condition.shape[0], -1, -1)
        tokens = torch.cat([condition, queries], dim=1)
        tokens = tokens + self.action_role.unsqueeze(0)
        tokens = self.action_transformer(tokens)
        action_tokens = tokens[:, 2 + self.subgoal_horizon :]
        mean = self.action_head(action_tokens)
        log_std = self.action_log_std_head(action_tokens)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        return mean, log_std

    def forward(self, z_t: torch.Tensor, z_g: torch.Tensor, z_h_teacher: torch.Tensor | None = None, teacher_force_subgoal: bool = False):
        leading_shape = z_t.shape[:-1]
        z_t = z_t.reshape(-1, self.latent_dim)
        z_g = z_g.reshape(-1, self.latent_dim)

        pred_z_h_seq = self.predict_subgoal(z_t, z_g)
        action_z_h_seq = (
            z_h_teacher.reshape(-1, self.subgoal_horizon, self.latent_dim)
            if teacher_force_subgoal and z_h_teacher is not None
            else pred_z_h_seq
        )
        action, action_log_std = self.predict_action_distribution(z_t, z_g, action_z_h_seq)

        pred_z_h_seq = pred_z_h_seq.reshape(*leading_shape, self.subgoal_horizon, self.latent_dim)
        action = action.reshape(*leading_shape, self.action_horizon, self.action_dim)
        action_log_std = action_log_std.reshape(*leading_shape, self.action_horizon, self.action_dim)
        return action, action_log_std, pred_z_h_seq
