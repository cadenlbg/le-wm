from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from latent_subgoal_act.wm_rollout import rollout_latent_with_actions


class LatentSubgoalACTWorldPolicy:
    PIXEL_KEYS: Sequence[str] = ("pixels", "obs", "observation")
    GOAL_KEYS: Sequence[str] = ("goal", "goal_pixels", "pixels_goal")

    def __init__(
        self,
        lewm_encoder: nn.Module,
        act_policy: nn.Module,
        *,
        transform: Optional[Dict[str, Any]] = None,
        action_mean: Optional[Any] = None,
        action_scale: Optional[Any] = None,
        device: str = "cuda",
        execute_horizon: int = 1,
        rerank_wm: Optional[nn.Module] = None,
        rerank_num_candidates: int = 1,
        rerank_noise_std: float = 0.0,
        rerank_target: str = "subgoal",
        cem_enabled: bool = False,
        cem_num_iters: int = 3,
        cem_num_candidates: int = 64,
        cem_elite_frac: float = 0.1,
        cem_init_std: float = 0.5,
        cem_min_std: float = 0.05,
        temporal_ensemble: bool = False,
        ensemble_decay: float = 0.01,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lewm_encoder = lewm_encoder.to(self.device).eval().requires_grad_(False)
        self.act_policy = act_policy.to(self.device).eval()
        self.transform = transform or {}
        self.action_mean = self._optional_tensor(action_mean)
        self.action_scale = self._optional_tensor(action_scale)
        self.execute_horizon = max(1, int(execute_horizon))
        self.rerank_wm = rerank_wm.to(self.device).eval().requires_grad_(False) if rerank_wm is not None else None
        self.rerank_num_candidates = max(1, int(rerank_num_candidates))
        self.rerank_noise_std = float(rerank_noise_std)
        self.rerank_target = rerank_target
        self.cem_enabled = bool(cem_enabled)
        self.cem_num_iters = max(1, int(cem_num_iters))
        self.cem_num_candidates = max(2, int(cem_num_candidates))
        self.cem_elite_frac = float(cem_elite_frac)
        self.cem_init_std = float(cem_init_std)
        self.cem_min_std = float(cem_min_std)
        self.temporal_ensemble = bool(temporal_ensemble)
        self.ensemble_decay = float(ensemble_decay)
        self.envs = None
        self._action_buffer = deque()
        self._chunk_history = deque()

    def set_env(self, envs):
        self.envs = envs

    def set_envs(self, envs):
        self.set_env(envs)

    def reset(self, *args, **kwargs):
        self._action_buffer.clear()
        self._chunk_history.clear()

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
        action_chunk, pred_z_h_seq = self.act_policy(z_t, z_g)
        action_chunk = self._maybe_plan_action_chunk(z_t, z_g, pred_z_h_seq, action_chunk)
        if self.temporal_ensemble:
            action_chunk = self._temporal_ensemble_action(action_chunk)
        action_chunk = self._inverse_action_scale(action_chunk).detach().cpu().numpy()
        self._fill_action_buffer(action_chunk)
        return self._action_buffer.popleft()

    def _maybe_plan_action_chunk(self, z_t: torch.Tensor, z_g: torch.Tensor, pred_z_h_seq: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        if self.rerank_wm is None:
            return action_chunk
        if self.cem_enabled:
            return self._cem_action_chunk(z_t, z_g, pred_z_h_seq, action_chunk)
        return self._maybe_rerank_action_chunk(z_t, z_g, pred_z_h_seq, action_chunk)

    def _temporal_ensemble_action(self, action_chunk: torch.Tensor) -> torch.Tensor:
        self._chunk_history.appendleft(action_chunk.detach())
        while len(self._chunk_history) > action_chunk.shape[1]:
            self._chunk_history.pop()

        actions = []
        weights = []
        for age, chunk in enumerate(self._chunk_history):
            if age >= chunk.shape[1]:
                continue
            actions.append(chunk[:, age])
            weights.append(torch.exp(action_chunk.new_tensor(-self.ensemble_decay * age)))

        stacked = torch.stack(actions, dim=1)
        weight = torch.stack(weights).view(1, -1, 1)
        action = (stacked * weight).sum(dim=1) / weight.sum()
        return action.unsqueeze(1)

    def _maybe_rerank_action_chunk(self, z_t: torch.Tensor, z_g: torch.Tensor, pred_z_h_seq: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        if self.rerank_wm is None or self.rerank_num_candidates <= 1:
            return action_chunk

        batch_size, horizon, action_dim = action_chunk.shape
        candidates = action_chunk.unsqueeze(1).expand(batch_size, self.rerank_num_candidates, horizon, action_dim).clone()
        if self.rerank_noise_std > 0 and self.rerank_num_candidates > 1:
            noise = torch.randn_like(candidates[:, 1:]) * self.rerank_noise_std
            candidates[:, 1:] = candidates[:, 1:] + noise

        flat_candidates = candidates.reshape(batch_size * self.rerank_num_candidates, horizon, action_dim)
        flat_z_t = z_t.unsqueeze(1).expand(batch_size, self.rerank_num_candidates, -1).reshape(batch_size * self.rerank_num_candidates, -1)
        rollout = rollout_latent_with_actions(self.rerank_wm, flat_z_t, flat_candidates)
        rollout = rollout.reshape(batch_size, self.rerank_num_candidates, -1)

        if self.rerank_target == "goal":
            target = z_g
        elif self.rerank_target == "subgoal":
            target = pred_z_h_seq[:, -1]
        else:
            raise ValueError("rerank_target must be 'subgoal' or 'goal'")

        cost = F.mse_loss(rollout, target.unsqueeze(1).expand_as(rollout), reduction="none").mean(dim=-1)
        best = cost.argmin(dim=1)
        batch_idx = torch.arange(batch_size, device=action_chunk.device)
        return candidates[batch_idx, best]

    def _cem_action_chunk(self, z_t: torch.Tensor, z_g: torch.Tensor, pred_z_h_seq: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, action_dim = action_chunk.shape
        mean = action_chunk.detach()
        std = torch.full_like(mean, self.cem_init_std)
        target = pred_z_h_seq[:, -1] if self.rerank_target == "subgoal" else z_g
        elite_count = max(1, int(round(self.cem_num_candidates * self.cem_elite_frac)))

        best_candidate = mean
        best_cost = torch.full((batch_size,), float("inf"), device=action_chunk.device)

        for _ in range(self.cem_num_iters):
            noise = torch.randn(
                batch_size,
                self.cem_num_candidates,
                horizon,
                action_dim,
                device=action_chunk.device,
                dtype=action_chunk.dtype,
            )
            candidates = mean.unsqueeze(1) + noise * std.unsqueeze(1)
            candidates[:, 0] = mean

            flat_candidates = candidates.reshape(batch_size * self.cem_num_candidates, horizon, action_dim)
            flat_z_t = z_t.unsqueeze(1).expand(batch_size, self.cem_num_candidates, -1).reshape(batch_size * self.cem_num_candidates, -1)
            rollout = rollout_latent_with_actions(self.rerank_wm, flat_z_t, flat_candidates)
            rollout = rollout.reshape(batch_size, self.cem_num_candidates, -1)
            cost = F.mse_loss(rollout, target.unsqueeze(1).expand_as(rollout), reduction="none").mean(dim=-1)

            iter_best_cost, iter_best_idx = cost.min(dim=1)
            improved = iter_best_cost < best_cost
            if improved.any():
                batch_idx = torch.arange(batch_size, device=action_chunk.device)
                best_candidate = torch.where(
                    improved.view(batch_size, 1, 1),
                    candidates[batch_idx, iter_best_idx],
                    best_candidate,
                )
                best_cost = torch.minimum(best_cost, iter_best_cost)

            elite_idx = torch.topk(cost, k=elite_count, dim=1, largest=False).indices
            gather_idx = elite_idx.view(batch_size, elite_count, 1, 1).expand(-1, -1, horizon, action_dim)
            elite = torch.gather(candidates, dim=1, index=gather_idx)
            mean = elite.mean(dim=1)
            std = elite.std(dim=1, unbiased=False).clamp_min(self.cem_min_std)

        return best_candidate

    def _encode_info_pixels(self, info: Dict[str, Any], keys: Iterable[str], transform_key: str) -> torch.Tensor:
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
        elif pixels.ndim != 5:
            raise ValueError(f"Unsupported pixel shape after transform: {tuple(pixels.shape)}")
        return pixels.to(self.device)

    @staticmethod
    def _apply_image_transform(pixels: Any, transform) -> torch.Tensor:
        array = LatentSubgoalACTWorldPolicy._as_numpy(pixels)
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
