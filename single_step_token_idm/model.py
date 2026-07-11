"""Single-step goal-conditioned token IDM.

The backbone follows the GC-IDM idea of a frozen latent encoder + goal-conditioned
inverse dynamics MLP, but the head predicts per-dimension categorical logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class TokenIDMConfig:
    """Model configuration."""

    embed_dim: int = 192
    action_dim: int = 2
    n_bins: int = 256
    hidden_dim: int = 512
    n_layers: int = 3
    dropout: float = 0.1
    activation: str = "gelu"
    noise_sigma: float = 0.0
    max_horizon: int = 50
    time_embed_dim: int = 64
    use_goal_delta: bool = True


class GoalConditionedTokenIDM(nn.Module):
    """Goal-conditioned inverse dynamics model with categorical action head."""

    def __init__(self, cfg: TokenIDMConfig):
        super().__init__()
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.n_bins = cfg.n_bins
        self.max_horizon = cfg.max_horizon

        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[cfg.activation]
        input_dim = cfg.embed_dim * 2 + (cfg.embed_dim if cfg.use_goal_delta else 0)

        self.horizon_embed = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.hidden_dim),
            act_fn(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(cfg.n_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, cfg.hidden_dim),
                    nn.LayerNorm(cfg.hidden_dim),
                    act_fn(),
                    nn.Dropout(cfg.dropout),
                ]
            )
            in_dim = cfg.hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.ada_scale = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.ada_shift = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.head = nn.Linear(cfg.hidden_dim, cfg.action_dim * cfg.n_bins)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.ada_scale.weight)
        nn.init.zeros_(self.ada_scale.bias)
        nn.init.zeros_(self.ada_shift.weight)
        nn.init.zeros_(self.ada_shift.bias)

    @staticmethod
    def _sinusoidal_embed(t: Tensor, dim: int = 64) -> Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def _apply_noise(self, z: Tensor) -> Tensor:
        if not self.training or self.cfg.noise_sigma <= 0:
            return z
        return z + self.cfg.noise_sigma * torch.randn_like(z)

    def forward(self, z_t: Tensor, z_goal: Tensor, steps_remaining: Tensor) -> Tensor:
        """Return logits with shape (B, action_dim, n_bins)."""
        z_t = self._apply_noise(z_t)
        z_goal = self._apply_noise(z_goal)
        goal_delta = z_goal - z_t if self.cfg.use_goal_delta else None

        h_frac = steps_remaining.float() / self.max_horizon
        h_emb = self.horizon_embed(self._sinusoidal_embed(h_frac, self.cfg.time_embed_dim))

        features = [z_t, z_goal]
        if goal_delta is not None:
            features.append(goal_delta)
        h = torch.cat(features, dim=-1)
        h = self.backbone(h)

        scale = self.ada_scale(h_emb)
        shift = self.ada_shift(h_emb)
        h = h * (1.0 + scale) + shift

        logits = self.head(h)
        return logits.view(-1, self.action_dim, self.n_bins)

    def predict_token_ids(
        self,
        logits: Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> Tensor:
        """Sample or decode token ids from logits."""
        if do_sample:
            probs = torch.softmax(logits / temperature, dim=-1)
            flat = probs.reshape(-1, probs.shape[-1])
            token_ids = torch.multinomial(flat, num_samples=1).view(logits.shape[0], logits.shape[1])
        else:
            token_ids = torch.argmax(logits, dim=-1)
        return token_ids

    def entropy(self, logits: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1).mean()

