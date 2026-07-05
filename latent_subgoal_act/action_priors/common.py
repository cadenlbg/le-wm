from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from latent_subgoal_act.shared import resolve_experiment_path


class GoalActionDataset(Dataset):
    def __init__(self, payload: Dict[str, Any], action_horizon: Optional[int] = None, max_samples: Optional[int] = None):
        required = ("z_t", "z_g", "action", "episode")
        for key in required:
            if key not in payload:
                raise KeyError(f"missing required key in payload: {key}")

        self.payload = payload
        available_horizon = int(payload["action"].shape[1])
        self.action_horizon = available_horizon if action_horizon is None else int(action_horizon)
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be >= 1")
        if self.action_horizon > available_horizon:
            raise ValueError(f"Requested action_horizon={self.action_horizon}, but dataset only has {available_horizon}.")
        self.length = int(payload["z_t"].shape[0])
        if max_samples is not None:
            self.length = min(self.length, int(max_samples))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "z_t": self.payload["z_t"][idx],
            "z_g": self.payload["z_g"][idx],
            "action": self.payload["action"][idx, : self.action_horizon],
            "episode": self.payload["episode"][idx],
        }


def episode_split(episodes: torch.Tensor, train_split: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    unique_episodes = torch.unique(episodes.cpu())
    perm = unique_episodes[torch.randperm(len(unique_episodes), generator=generator)]
    n_train = max(1, int(len(perm) * float(train_split)))
    train_eps = set(perm[:n_train].tolist())
    train_idx, val_idx = [], []
    for idx, episode in enumerate(episodes.tolist()):
        (train_idx if episode in train_eps else val_idx).append(idx)
    if not val_idx:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    return train_idx, val_idx


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def save_policy_checkpoint(path: Path, model, cfg, metadata: Dict[str, Any], epoch: int, val_score: float, extra_config: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": extra_config,
            "metadata": metadata,
            "epoch": epoch,
            "val_score": val_score,
            "config": cfg,
        },
        path,
    )


def resolve_policy_ckpt(path) -> Path:
    return resolve_experiment_path(path)


class PixelEncoderMixin:
    PIXEL_KEYS: Sequence[str] = ("pixels", "obs", "observation")
    GOAL_KEYS: Sequence[str] = ("goal", "goal_pixels", "pixels_goal")

    @staticmethod
    def first_present(mapping: Dict[str, Any], keys: Iterable[str]):
        for key in keys:
            if key in mapping:
                return mapping[key]
        raise KeyError(f"None of these keys were found in policy info: {tuple(keys)}")

    @staticmethod
    def as_tensor(value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value
        return torch.as_tensor(np.asarray(value))

    @staticmethod
    def as_numpy(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def apply_image_transform(cls, pixels: Any, transform) -> torch.Tensor:
        array = cls.as_numpy(pixels)
        if array.ndim == 3:
            return transform(array)
        if array.ndim < 4:
            raise ValueError(f"Unsupported pixel shape before transform: {array.shape}")
        leading_shape = array.shape[:-3]
        frames = array.reshape(-1, *array.shape[-3:])
        frames = torch.stack([transform(frame) for frame in frames], dim=0)
        return frames.reshape(*leading_shape, *frames.shape[-3:])

    def prepare_pixels(self, pixels: Any, transform_key: str) -> torch.Tensor:
        transform = self.transform.get(transform_key)
        if transform is not None:
            pixels = self.apply_image_transform(pixels, transform)
        else:
            pixels = self.as_tensor(pixels)

        pixels = pixels.float()
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0).unsqueeze(0)
        elif pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        elif pixels.ndim != 5:
            raise ValueError(f"Unsupported pixel shape after transform: {tuple(pixels.shape)}")
        return pixels.to(self.device)

    @torch.no_grad()
    def encode_info_pixels(self, info: Dict[str, Any], keys: Iterable[str], transform_key: str) -> torch.Tensor:
        pixels = self.first_present(info, keys)
        pixels = self.prepare_pixels(pixels, transform_key)
        output = self.lewm_encoder.encode({"pixels": pixels})
        return output["emb"][:, -1]

