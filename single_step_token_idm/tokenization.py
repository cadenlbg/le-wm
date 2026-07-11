"""Action tokenization helpers.

This module uses fixed binning, following the spirit of SimpleVLA-RL:
continuous actions are normalized to [-1, 1], discretized into bins, and
decoded by bin centers. There is no learned tokenizer here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

import numpy as np
import torch


NormalizationMode = Literal["bounds", "bounds_q99"]


@dataclass
class ActionTokenizerConfig:
    """Configuration for fixed action discretization."""

    action_dim: int = 2
    n_bins: int = 256
    normalization: NormalizationMode = "bounds_q99"
    token_offset: int = 0
    eps: float = 1e-8


@dataclass
class ActionStats:
    """Dataset-derived action statistics."""

    action_min: np.ndarray
    action_max: np.ndarray
    action_q01: Optional[np.ndarray] = None
    action_q99: Optional[np.ndarray] = None
    action_mask: Optional[np.ndarray] = None


def compute_action_stats(actions: np.ndarray, use_q99: bool = True) -> ActionStats:
    """Compute action bounds from a raw action array."""
    actions = np.asarray(actions, dtype=np.float32)
    valid_rows = np.isfinite(actions).all(axis=1)
    if not np.any(valid_rows):
        raise ValueError("No finite action rows available for tokenizer statistics.")

    finite_actions = actions[valid_rows]
    action_min = np.min(finite_actions, axis=0)
    action_max = np.max(finite_actions, axis=0)
    action_q01 = np.quantile(finite_actions, 0.01, axis=0) if use_q99 else None
    action_q99 = np.quantile(finite_actions, 0.99, axis=0) if use_q99 else None
    action_mask = np.isfinite(finite_actions).all(axis=0)
    return ActionStats(
        action_min=action_min,
        action_max=action_max,
        action_q01=action_q01,
        action_q99=action_q99,
        action_mask=action_mask,
    )


class ActionTokenizer:
    """Fixed bin action tokenizer / detokenizer."""

    def __init__(
        self,
        cfg: ActionTokenizerConfig,
        action_low: np.ndarray,
        action_high: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.cfg = cfg
        self.action_dim = int(cfg.action_dim)
        self.n_bins = int(cfg.n_bins)
        self.token_offset = int(cfg.token_offset)
        self.eps = float(cfg.eps)
        self.action_low = self._as_action_array(action_low)
        self.action_high = self._as_action_array(action_high)
        self.action_mask = None if action_mask is None else np.asarray(action_mask, dtype=bool)

    @classmethod
    def from_stats(cls, stats: ActionStats, cfg: ActionTokenizerConfig) -> "ActionTokenizer":
        if cfg.normalization == "bounds_q99" and stats.action_q01 is not None and stats.action_q99 is not None:
            low = stats.action_q01
            high = stats.action_q99
        else:
            low = stats.action_min
            high = stats.action_max
        return cls(cfg=cfg, action_low=low, action_high=high, action_mask=stats.action_mask)

    @staticmethod
    def _as_action_array(value: np.ndarray | float | int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full((1,), float(arr), dtype=np.float32)
        return arr

    @property
    def bin_edges(self) -> np.ndarray:
        return np.linspace(-1.0, 1.0, self.n_bins + 1, dtype=np.float32)

    @property
    def bin_centers(self) -> np.ndarray:
        edges = self.bin_edges
        return (edges[:-1] + edges[1:]) / 2.0

    def to_dict(self) -> dict:
        return {
            "cfg": asdict(self.cfg),
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "action_mask": None if self.action_mask is None else self.action_mask.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActionTokenizer":
        cfg = ActionTokenizerConfig(**data["cfg"])
        return cls(
            cfg=cfg,
            action_low=np.asarray(data["action_low"], dtype=np.float32),
            action_high=np.asarray(data["action_high"], dtype=np.float32),
            action_mask=None if data.get("action_mask") is None else np.asarray(data["action_mask"], dtype=bool),
        )

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        low = np.broadcast_to(self.action_low, actions.shape)
        high = np.broadcast_to(self.action_high, actions.shape)
        normalized = 2.0 * (actions - low) / (high - low + self.eps) - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)
        if self.action_mask is not None:
            mask = np.broadcast_to(self.action_mask, actions.shape)
            normalized = np.where(mask, normalized, actions)
        return normalized

    def denormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        low = np.broadcast_to(self.action_low, normalized_actions.shape)
        high = np.broadcast_to(self.action_high, normalized_actions.shape)
        denormalized = 0.5 * (normalized_actions + 1.0) * (high - low + self.eps) + low
        if self.action_mask is not None:
            mask = np.broadcast_to(self.action_mask, normalized_actions.shape)
            denormalized = np.where(mask, denormalized, normalized_actions)
        return denormalized

    def actions_to_token_ids(self, actions: np.ndarray) -> np.ndarray:
        """Map continuous actions to discrete token ids."""
        normalized = self.normalize_actions(actions)
        scaled = (normalized + 1.0) * 0.5 * (self.n_bins - 1)
        bin_ids = np.rint(scaled).astype(np.int64)
        bin_ids = np.clip(bin_ids, 0, self.n_bins - 1)
        return bin_ids + self.token_offset

    def token_ids_to_actions(self, token_ids: np.ndarray) -> np.ndarray:
        """Map discrete token ids back to continuous actions."""
        token_ids = np.asarray(token_ids, dtype=np.int64) - self.token_offset
        token_ids = np.clip(token_ids, 0, self.n_bins - 1)
        normalized = self.bin_centers[token_ids]
        return self.denormalize_actions(normalized)

    def torch_bin_centers(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(self.bin_centers, device=device, dtype=dtype)

    def expected_actions_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute action expectation from categorical logits.

        Args:
            logits: (B, action_dim, n_bins)
        """
        probs = torch.softmax(logits, dim=-1)
        centers = self.torch_bin_centers(logits.device, logits.dtype)
        normalized = torch.sum(probs * centers.view(1, 1, -1), dim=-1)
        low = torch.as_tensor(self.action_low, device=logits.device, dtype=logits.dtype)
        high = torch.as_tensor(self.action_high, device=logits.device, dtype=logits.dtype)
        low = low.view(1, -1)
        high = high.view(1, -1)
        return 0.5 * (normalized + 1.0) * (high - low + self.eps) + low
