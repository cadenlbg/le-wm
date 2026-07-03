from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn


class LatentGoalBCPolicy(nn.Module):
    """MLP policy that predicts an action chunk from LeWM goal-conditioned latents."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int = 512,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        layers = []
        input_dim = 3 * latent_dim
        for layer_idx in range(depth):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, action_horizon * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z_t: torch.Tensor,
        z_g: torch.Tensor,
        delta_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if delta_z is None:
            delta_z = z_g - z_t
        x = torch.cat([z_t, z_g, delta_z], dim=-1)
        action = self.net(x)
        return action.view(*action.shape[:-1], self.action_horizon, self.action_dim)


class LatentBCWorldPolicy:
    """Runtime policy wrapper that replaces CEM with a latent BC action predictor."""

    def __init__(
        self,
        lewm_encoder: nn.Module,
        bc_policy: LatentGoalBCPolicy,
        *,
        transform: Optional[Dict[str, Any]] = None,
        action_mean: Optional[Any] = None,
        action_scale: Optional[Any] = None,
        device: str = "cuda",
        execute_horizon: int = 1,
    ):
        self.lewm_encoder = lewm_encoder
        self.bc_policy = bc_policy
        self.transform = transform or {}
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.execute_horizon = execute_horizon
        self._action_buffer = deque()

        self.lewm_encoder.to(self.device).eval().requires_grad_(False)
        self.bc_policy.to(self.device).eval()

        self.action_mean = self._as_tensor(action_mean)
        self.action_scale = self._as_tensor(action_scale)

    def reset(self):
        self._action_buffer.clear()

    def __call__(self, info: Dict[str, Any]):
        return self.act(info)

    def get_action(self, info: Dict[str, Any]):
        return self.act(info)

    @torch.no_grad()
    def act(self, info: Dict[str, Any]):
        if self._action_buffer:
            return self._action_buffer.popleft()

        z_t = self._encode_pixels(info["pixels"], key="pixels")
        z_g = self._encode_pixels(info["goal"], key="goal")
        action = self.bc_policy(z_t, z_g)[0]
        action = self._inverse_action_scale(action).detach().cpu().numpy()

        n_to_buffer = min(max(self.execute_horizon, 1), action.shape[0])
        for idx in range(n_to_buffer):
            self._action_buffer.append(action[idx])
        return self._action_buffer.popleft()

    def _as_tensor(self, value):
        if value is None:
            return None
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _inverse_action_scale(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_mean is None or self.action_scale is None:
            return action
        return action * self.action_scale + self.action_mean

    def _encode_pixels(self, pixels: Any, key: str) -> torch.Tensor:
        pixels = self._prepare_pixels(pixels, key)
        output = self.lewm_encoder.encode({"pixels": pixels})
        emb = output["emb"]
        return emb[:, -1]

    def _prepare_pixels(self, pixels: Any, key: str) -> torch.Tensor:
        if not torch.is_tensor(pixels):
            pixels = torch.as_tensor(np.asarray(pixels))

        transform = self.transform.get(key)
        if transform is not None:
            if pixels.ndim >= 4:
                flat = pixels.reshape(-1, *pixels.shape[-3:])
                flat = torch.stack([transform(img) for img in flat], dim=0)
                pixels = flat.reshape(*pixels.shape[:-3], *flat.shape[-3:])
            else:
                pixels = transform(pixels)

        pixels = pixels.float()
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0).unsqueeze(0)
        elif pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        elif pixels.ndim == 5:
            pass
        elif pixels.ndim == 6:
            pixels = pixels.reshape(-1, *pixels.shape[2:])
        else:
            raise ValueError(f"Unsupported pixel shape: {tuple(pixels.shape)}")
        return pixels.to(self.device)
