from collections import deque
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from torch import nn


class LatentGoalBCPolicy(nn.Module):
    """Predict an action chunk from current and goal LeWM latents."""

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

        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)

        layers = []
        input_dim = 3 * self.latent_dim
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
        layers.append(nn.Linear(hidden_dim, self.action_horizon * self.action_dim))
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
        return action.reshape(*action.shape[:-1], self.action_horizon, self.action_dim)


class LatentBCWorldPolicy:
    """stable_worldmodel-compatible wrapper for a frozen LeWM encoder plus BC policy."""

    PIXEL_KEYS: Sequence[str] = ("pixels", "obs", "observation")
    GOAL_KEYS: Sequence[str] = ("goal", "goal_pixels", "pixels_goal")

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
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lewm_encoder = lewm_encoder.to(self.device).eval().requires_grad_(False)
        self.bc_policy = bc_policy.to(self.device).eval()

        self.transform = transform or {}
        self.execute_horizon = max(1, int(execute_horizon))
        self.action_mean = self._optional_tensor(action_mean)
        self.action_scale = self._optional_tensor(action_scale)

        self.envs = None
        self._action_buffer = deque()

    # stable_worldmodel policy compatibility.
    def set_env(self, envs):
        self.envs = envs

    def set_envs(self, envs):
        self.set_env(envs)

    def reset(self, *args, **kwargs):
        self._action_buffer.clear()

    def __call__(self, info: Dict[str, Any]):
        return self.act(info)

    def get_action(self, info: Dict[str, Any]):
        return self.act(info)

    @torch.no_grad()
    def act(self, info: Dict[str, Any]):
        if self._action_buffer:
            return self._action_buffer.popleft()

        z_t = self._encode_info_pixels(info, self.PIXEL_KEYS, "pixels")
        z_g = self._encode_info_pixels(info, self.GOAL_KEYS, "goal")

        action_chunk = self.bc_policy(z_t, z_g)
        action_chunk = self._inverse_action_scale(action_chunk)
        action_chunk = action_chunk.detach().cpu().numpy()

        self._fill_action_buffer(action_chunk)
        return self._action_buffer.popleft()

    def _encode_info_pixels(
        self,
        info: Dict[str, Any],
        keys: Iterable[str],
        transform_key: str,
    ) -> torch.Tensor:
        pixels = self._first_present(info, keys)
        pixels = self._prepare_pixels(pixels, transform_key)
        output = self.lewm_encoder.encode({"pixels": pixels})
        return output["emb"][:, -1]

    def _prepare_pixels(self, pixels: Any, transform_key: str) -> torch.Tensor:
        transform = self.transform.get(transform_key)
        if transform is not None:
            pixels = self._apply_image_transform(pixels, transform)
        else:
            pixels = self._as_tensor(pixels)

        pixels = pixels.float()
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0).unsqueeze(0)
        elif pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        elif pixels.ndim == 5:
            pass
        else:
            raise ValueError(f"Unsupported pixel shape after transform: {tuple(pixels.shape)}")
        return pixels.to(self.device)

    @staticmethod
    def _apply_image_transform(pixels: Any, transform) -> torch.Tensor:
        array = LatentBCWorldPolicy._as_numpy(pixels)

        if array.ndim == 3:
            return transform(array)

        if array.ndim < 4:
            raise ValueError(f"Unsupported pixel shape before transform: {array.shape}")

        leading_shape = array.shape[:-3]
        frames = array.reshape(-1, *array.shape[-3:])
        frames = torch.stack([transform(frame) for frame in frames], dim=0)
        return frames.reshape(*leading_shape, *frames.shape[-3:])

    def _fill_action_buffer(self, action_chunk: np.ndarray):
        horizon = min(self.execute_horizon, action_chunk.shape[1])
        batched = action_chunk.shape[0] > 1

        for step in range(horizon):
            action = action_chunk[:, step, :] if batched else action_chunk[0, step, :]
            self._action_buffer.append(action)

    def _inverse_action_scale(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_mean is None or self.action_scale is None:
            return action
        return action * self.action_scale + self.action_mean

    def _optional_tensor(self, value):
        if value is None:
            return None
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    @staticmethod
    def _as_tensor(value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value
        return torch.as_tensor(np.asarray(value))

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _first_present(mapping: Dict[str, Any], keys: Iterable[str]):
        for key in keys:
            if key in mapping:
                return mapping[key]
        raise KeyError(f"None of these keys were found in policy info: {tuple(keys)}")
