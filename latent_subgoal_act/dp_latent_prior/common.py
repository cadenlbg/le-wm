from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from latent_subgoal_act.action_priors.common import GoalActionDataset, episode_split, move_batch, resolve_policy_ckpt


class ActionNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        self.mean = mean
        self.std = std.clamp_min(eps)

    @classmethod
    def from_actions(cls, actions: torch.Tensor, enabled: bool = True):
        if not enabled:
            mean = torch.zeros(actions.shape[-1], dtype=actions.dtype)
            std = torch.ones(actions.shape[-1], dtype=actions.dtype)
            return cls(mean, std)
        flat = actions.reshape(-1, actions.shape[-1]).float()
        return cls(flat.mean(dim=0), flat.std(dim=0, unbiased=False))

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.mean.view(1, 1, -1)) / self.std.view(1, 1, -1)

    def unnormalize(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.std.view(1, 1, -1) + self.mean.view(1, 1, -1)

    def state_dict(self):
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu()}

    @classmethod
    def load(cls, state):
        return cls(torch.as_tensor(state["mean"]).float(), torch.as_tensor(state["std"]).float())


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.model = deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        for ema_buffer, buffer in zip(self.model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)

    def to(self, device):
        self.model.to(device)
        return self


def save_dp_checkpoint(
    path: Path,
    model: nn.Module,
    ema_model: Optional[nn.Module],
    cfg,
    metadata: Dict[str, Any],
    epoch: int,
    val_score: float,
    model_config: Dict[str, Any],
    normalizer: ActionNormalizer,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict() if ema_model is not None else None,
            "model_config": model_config,
            "metadata": metadata,
            "normalizer": normalizer.state_dict(),
            "epoch": epoch,
            "val_score": val_score,
            "config": cfg,
        },
        path,
    )

