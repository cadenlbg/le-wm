"""Autoregressive K-step goal-conditioned token IDM.

The condition encoder follows the single-step token IDM design, except that it
uses only ``[z_t, z_goal]``. A causal Transformer then predicts a categorical
distribution for each action dimension at each step of an action trunk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class KStepTokenIDMConfig:
    """Model configuration."""

    embed_dim: int = 192
    action_dim: int = 2
    n_bins: int = 256
    action_horizon: int = 3
    hidden_dim: int = 512
    condition_layers: int = 3
    transformer_layers: int = 3
    transformer_heads: int = 8
    transformer_ffn_dim: int = 2048
    dropout: float = 0.1
    activation: str = "gelu"
    noise_sigma: float = 0.0
    max_goal_horizon: int = 50
    time_embed_dim: int = 64
    token_offset: int = 0


class AutoregressiveKStepTokenIDM(nn.Module):
    """Predict an autoregressive categorical distribution over an action trunk."""

    def __init__(self, cfg: KStepTokenIDMConfig):
        super().__init__()
        self._validate_config(cfg)
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.n_bins = cfg.n_bins
        self.action_horizon = cfg.action_horizon
        self.max_horizon = cfg.max_goal_horizon
        self.token_offset = cfg.token_offset

        act_cls = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[cfg.activation]
        transformer_activation = {
            "gelu": F.gelu,
            "relu": F.relu,
            "silu": F.silu,
        }[cfg.activation]

        self.horizon_embed = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.hidden_dim),
            act_cls(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        condition: list[nn.Module] = []
        in_dim = cfg.embed_dim * 2
        for _ in range(cfg.condition_layers):
            condition.extend(
                [
                    nn.Linear(in_dim, cfg.hidden_dim),
                    nn.LayerNorm(cfg.hidden_dim),
                    act_cls(),
                    nn.Dropout(cfg.dropout),
                ]
            )
            in_dim = cfg.hidden_dim
        self.condition_backbone = nn.Sequential(*condition)
        self.ada_scale = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.ada_shift = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.action_embeddings = nn.ModuleList(
            [nn.Embedding(cfg.n_bins, cfg.hidden_dim) for _ in range(cfg.action_dim)]
        )
        self.bos_token = nn.Parameter(torch.empty(1, 1, cfg.hidden_dim))
        self.position_embedding = nn.Parameter(
            torch.empty(1, cfg.action_horizon, cfg.hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ffn_dim,
            dropout=cfg.dropout,
            activation=transformer_activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.transformer_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(cfg.hidden_dim)
        self.action_heads = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, cfg.n_bins) for _ in range(cfg.action_dim)]
        )
        self._init_weights()

    @staticmethod
    def _validate_config(cfg: KStepTokenIDMConfig) -> None:
        if cfg.action_dim < 1 or cfg.n_bins < 2 or cfg.action_horizon < 1:
            raise ValueError("action_dim, n_bins, and action_horizon must be positive")
        if cfg.condition_layers < 1 or cfg.transformer_layers < 1:
            raise ValueError("condition_layers and transformer_layers must be positive")
        if cfg.hidden_dim % cfg.transformer_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        if cfg.time_embed_dim % 2 != 0:
            raise ValueError("time_embed_dim must be even")
        if cfg.activation not in {"gelu", "relu", "silu"}:
            raise ValueError("activation must be 'gelu', 'relu', or 'silu'")

    def _init_weights(self) -> None:
        condition_modules = [self.horizon_embed, self.condition_backbone]
        for parent in condition_modules:
            for module in parent.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        nn.init.zeros_(self.ada_scale.weight)
        nn.init.zeros_(self.ada_scale.bias)
        nn.init.zeros_(self.ada_shift.weight)
        nn.init.zeros_(self.ada_shift.bias)
        nn.init.normal_(self.bos_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        for embedding in self.action_embeddings:
            nn.init.normal_(embedding.weight, std=0.02)
        for head in self.action_heads:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _sinusoidal_embed(t: Tensor, dim: int) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def _apply_noise(self, z: Tensor) -> Tensor:
        if not self.training or self.cfg.noise_sigma <= 0:
            return z
        return z + self.cfg.noise_sigma * torch.randn_like(z)

    def encode_condition(
        self,
        z_t: Tensor,
        z_goal: Tensor,
        steps_remaining: Tensor,
    ) -> Tensor:
        """Encode ``[z_t, z_goal]`` with the original horizon modulation."""
        self._validate_condition_inputs(z_t, z_goal, steps_remaining)
        z_t = self._apply_noise(z_t)
        z_goal = self._apply_noise(z_goal)
        h = self.condition_backbone(torch.cat([z_t, z_goal], dim=-1))

        horizon_fraction = steps_remaining.float() / self.max_horizon
        horizon = self._sinusoidal_embed(
            horizon_fraction, self.cfg.time_embed_dim
        )
        horizon = self.horizon_embed(horizon)
        return h * (1.0 + self.ada_scale(horizon)) + self.ada_shift(horizon)

    def forward(
        self,
        z_t: Tensor,
        z_goal: Tensor,
        steps_remaining: Tensor,
        target_tokens: Tensor,
    ) -> Tensor:
        """Teacher-forced logits with shape ``(B, K, action_dim, n_bins)``.

        ``target_tokens`` contains the full ground-truth action trunk. The model
        shifts it internally, so the hidden state predicting step ``i`` can only
        attend to condition information and action steps before ``i``.
        """
        self._validate_target_tokens(target_tokens, z_t.shape[0])
        condition = self.encode_condition(z_t, z_goal, steps_remaining)
        previous_tokens = target_tokens[:, :-1]
        hidden = self._decode_hidden(condition, previous_tokens)
        return self._project_logits(hidden)

    @torch.no_grad()
    def generate(
        self,
        z_t: Tensor,
        z_goal: Tensor,
        steps_remaining: Tensor,
        *,
        horizon: Optional[int] = None,
        num_samples: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """Generate token trunks with shape ``(B, num_samples, K, action_dim)``."""
        horizon = self.action_horizon if horizon is None else int(horizon)
        if not 1 <= horizon <= self.action_horizon:
            raise ValueError("horizon must be in [1, action_horizon]")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and not 1 <= top_k <= self.n_bins:
            raise ValueError("top_k must be in [1, n_bins]")

        condition = self.encode_condition(z_t, z_goal, steps_remaining)
        batch_size = condition.shape[0]
        condition = condition.repeat_interleave(num_samples, dim=0)
        generated = torch.empty(
            batch_size * num_samples,
            0,
            self.action_dim,
            device=z_t.device,
            dtype=torch.long,
        )

        for _ in range(horizon):
            hidden = self._decode_hidden(condition, generated)
            logits = self._project_logits(hidden[:, -1:])[:, 0]
            next_tokens = self._decode_next_tokens(
                logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
            ) + self.token_offset
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

        return generated.view(
            batch_size, num_samples, horizon, self.action_dim
        )

    def _decode_hidden(self, condition: Tensor, previous_tokens: Tensor) -> Tensor:
        """Return one hidden state for BOS and each supplied previous action."""
        batch_size, history_length, _ = previous_tokens.shape
        sequence_length = history_length + 1
        if sequence_length > self.action_horizon:
            raise ValueError("action history exceeds action_horizon")

        bos = self.bos_token.expand(batch_size, -1, -1)
        if history_length:
            bin_ids = self._tokens_to_bin_ids(previous_tokens)
            action_embedding = sum(
                embedding(bin_ids[:, :, dim])
                for dim, embedding in enumerate(self.action_embeddings)
            ) / math.sqrt(self.action_dim)
            action_inputs = torch.cat([bos, action_embedding], dim=1)
        else:
            action_inputs = bos

        action_inputs = action_inputs + self.position_embedding[:, :sequence_length]
        sequence = torch.cat([condition.unsqueeze(1), action_inputs], dim=1)
        mask = torch.triu(
            torch.ones(
                sequence.shape[1],
                sequence.shape[1],
                device=sequence.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        hidden = self.transformer(sequence, mask=mask)
        return self.final_norm(hidden[:, 1:])

    def _project_logits(self, hidden: Tensor) -> Tensor:
        return torch.stack([head(hidden) for head in self.action_heads], dim=2)

    def _decode_next_tokens(
        self,
        logits: Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int],
    ) -> Tensor:
        logits = logits / temperature
        if top_k is not None:
            threshold = torch.topk(logits, k=top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        if not do_sample:
            return torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        flat = probabilities.reshape(-1, self.n_bins)
        return torch.multinomial(flat, num_samples=1).view(
            probabilities.shape[0], self.action_dim
        )

    def _tokens_to_bin_ids(self, tokens: Tensor) -> Tensor:
        bin_ids = tokens.long() - self.token_offset
        if torch.any(bin_ids < 0) or torch.any(bin_ids >= self.n_bins):
            raise ValueError("action token ids are outside the configured vocabulary")
        return bin_ids

    def _validate_condition_inputs(
        self,
        z_t: Tensor,
        z_goal: Tensor,
        steps_remaining: Tensor,
    ) -> None:
        expected = (z_t.shape[0], self.cfg.embed_dim)
        if z_t.ndim != 2 or tuple(z_t.shape) != expected:
            raise ValueError(f"z_t must have shape {expected}")
        if tuple(z_goal.shape) != expected:
            raise ValueError(f"z_goal must have shape {expected}")
        if tuple(steps_remaining.shape) != (z_t.shape[0],):
            raise ValueError("steps_remaining must have shape (B,)")

    def _validate_target_tokens(self, tokens: Tensor, batch_size: int) -> None:
        expected = (batch_size, self.action_horizon, self.action_dim)
        if tuple(tokens.shape) != expected:
            raise ValueError(f"target_tokens must have shape {expected}")
        self._tokens_to_bin_ids(tokens)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the model parameter count."""
        parameters = (p for p in self.parameters() if p.requires_grad) if trainable_only else self.parameters()
        return sum(parameter.numel() for parameter in parameters)
