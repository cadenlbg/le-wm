"""Episode-safe K-step action-trunk dataset over frozen LeWM embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from single_step_token_idm.tokenization import ActionStats, ActionTokenizer, compute_action_stats

from .splits import EpisodeSplitManifest, Partition


GoalSampling = Literal["fixed", "uniform"]


class EmbeddingStore:
    """Load one embedding archive and share it across dataset partitions."""

    def __init__(self, path: str | Path):
        self.path = str(Path(path).resolve())
        payload = np.load(path, allow_pickle=True)
        required = {"embeddings", "actions", "episode_ids"}
        missing = required - set(payload.files)
        if missing:
            raise KeyError(f"embedding archive is missing keys: {sorted(missing)}")
        self.embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        self.actions = np.asarray(payload["actions"], dtype=np.float32)
        self.episode_ids = np.asarray(payload["episode_ids"], dtype=np.int64)
        if not (len(self.embeddings) == len(self.actions) == len(self.episode_ids)):
            raise ValueError("embeddings, actions, and episode_ids must have equal length")
        if self.embeddings.ndim != 2 or self.actions.ndim != 2:
            raise ValueError("embeddings and actions must both be rank-2 arrays")
        if len(self.episode_ids) == 0:
            raise ValueError("embedding archive is empty")
        self.episode_ranges = self._build_episode_ranges()

    @property
    def unique_episode_ids(self) -> np.ndarray:
        return np.asarray(sorted(self.episode_ranges), dtype=np.int64)

    def action_stats(self, episode_ids: list[int], use_q99: bool = True) -> ActionStats:
        mask = np.isin(self.episode_ids, np.asarray(episode_ids, dtype=np.int64))
        return compute_action_stats(self.actions[mask], use_q99=use_q99)

    def _build_episode_ranges(self) -> dict[int, tuple[int, int]]:
        ranges: dict[int, tuple[int, int]] = {}
        start = 0
        current = int(self.episode_ids[0])
        for index in range(1, len(self.episode_ids)):
            episode = int(self.episode_ids[index])
            if episode == current:
                continue
            if current in ranges:
                raise ValueError(f"episode {current} appears in non-contiguous segments")
            ranges[current] = (start, index)
            start = index
            current = episode
        if current in ranges:
            raise ValueError(f"episode {current} appears in non-contiguous segments")
        ranges[current] = (start, len(self.episode_ids))
        return ranges


class KStepEmbeddingDataset(Dataset):
    """Return ``(z_t, z_goal, G, K-step actions/tokens)`` samples."""

    def __init__(
        self,
        store: EmbeddingStore | str | Path,
        manifest: EpisodeSplitManifest,
        partition: Partition,
        *,
        action_horizon: int = 3,
        goal_offset: int = 25,
        goal_sampling: GoalSampling = "fixed",
        max_goal_horizon: int = 50,
        tokenizer: Optional[ActionTokenizer] = None,
        goal_seed: int = 0,
    ):
        self.store = store if isinstance(store, EmbeddingStore) else EmbeddingStore(store)
        manifest.validate(self.store.unique_episode_ids)
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if goal_offset < action_horizon:
            raise ValueError("goal_offset must be at least action_horizon")
        if max_goal_horizon < action_horizon:
            raise ValueError("max_goal_horizon must be at least action_horizon")
        if goal_sampling == "fixed" and goal_offset > max_goal_horizon:
            raise ValueError("fixed goal_offset cannot exceed max_goal_horizon")
        if goal_sampling not in {"fixed", "uniform"}:
            raise ValueError("goal_sampling must be 'fixed' or 'uniform'")

        self.manifest = manifest
        self.partition = partition
        self.action_horizon = int(action_horizon)
        self.goal_offset = int(goal_offset)
        self.goal_sampling = goal_sampling
        self.max_goal_horizon = int(max_goal_horizon)
        self.tokenizer = tokenizer
        self.goal_seed = int(goal_seed)
        self.epoch = 0
        self.valid_indices = self._build_valid_indices()
        if not self.valid_indices:
            raise ValueError(f"partition {partition!r} has no valid K-step samples")

    @property
    def embeddings(self) -> np.ndarray:
        return self.store.embeddings

    @property
    def actions(self) -> np.ndarray:
        return self.store.actions

    def set_tokenizer(self, tokenizer: ActionTokenizer) -> None:
        self.tokenizer = tokenizer

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_valid_indices(self) -> list[int]:
        valid: list[int] = []
        for episode in self.manifest.episode_ids(self.partition):
            start, end = self.store.episode_ranges[episode]
            required_future = self.goal_offset if self.goal_sampling == "fixed" else self.action_horizon
            last_start = end - 1 - required_future
            for index in range(start, last_start + 1):
                action_chunk = self.store.actions[index : index + self.action_horizon]
                if action_chunk.shape[0] == self.action_horizon and np.isfinite(action_chunk).all():
                    valid.append(index)
        return valid

    def _goal_index(self, index: int, dataset_index: int) -> int:
        if self.goal_sampling == "fixed":
            return index + self.goal_offset
        episode = int(self.store.episode_ids[index])
        _, end = self.store.episode_ranges[episode]
        maximum = min(index + self.max_goal_horizon, end - 1)
        seed = self.goal_seed + self.epoch * len(self.valid_indices) + dataset_index
        rng = np.random.default_rng(seed)
        return int(rng.integers(index + self.action_horizon, maximum + 1))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, dataset_index: int) -> dict[str, torch.Tensor]:
        index = self.valid_indices[dataset_index]
        goal_index = self._goal_index(index, dataset_index)
        episode = int(self.store.episode_ids[index])
        actions = self.store.actions[index : index + self.action_horizon]
        sample = {
            "z_t": torch.from_numpy(self.store.embeddings[index]).float(),
            "z_goal": torch.from_numpy(self.store.embeddings[goal_index]).float(),
            "steps_remaining": torch.tensor(goal_index - index, dtype=torch.long),
            "actions": torch.from_numpy(actions).float(),
            "episode_id": torch.tensor(episode, dtype=torch.long),
            "start_index": torch.tensor(index, dtype=torch.long),
            "goal_index": torch.tensor(goal_index, dtype=torch.long),
        }
        if self.tokenizer is not None:
            token_ids = self.tokenizer.actions_to_token_ids(actions)
            sample["action_tokens"] = torch.from_numpy(token_ids).long()
        return sample
