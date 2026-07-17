"""Deterministic episode-level train/validation/test splits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np


Partition = Literal["train", "val", "test"]


@dataclass(frozen=True)
class EpisodeSplitManifest:
    """Serializable episode partition manifest."""

    split_seed: int
    test_fraction: float
    val_fraction_of_remaining: float
    train_episode_ids: list[int]
    val_episode_ids: list[int]
    test_episode_ids: list[int]

    def episode_ids(self, partition: Partition) -> list[int]:
        return {
            "train": self.train_episode_ids,
            "val": self.val_episode_ids,
            "test": self.test_episode_ids,
        }[partition]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "EpisodeSplitManifest":
        manifest = cls(
            split_seed=int(payload["split_seed"]),
            test_fraction=float(payload["test_fraction"]),
            val_fraction_of_remaining=float(payload["val_fraction_of_remaining"]),
            train_episode_ids=[int(x) for x in payload["train_episode_ids"]],
            val_episode_ids=[int(x) for x in payload["val_episode_ids"]],
            test_episode_ids=[int(x) for x in payload["test_episode_ids"]],
        )
        manifest.validate()
        return manifest

    def validate(self, expected_episode_ids: Iterable[int] | None = None) -> None:
        groups = [
            set(self.train_episode_ids),
            set(self.val_episode_ids),
            set(self.test_episode_ids),
        ]
        if any(not group for group in groups):
            raise ValueError("train, val, and test episode partitions must be non-empty")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("episode partitions overlap")
        if expected_episode_ids is not None:
            expected = {int(x) for x in expected_episode_ids}
            actual = set.union(*groups)
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise ValueError(
                    f"split manifest does not match dataset episodes; missing={missing}, extra={extra}"
                )


def create_episode_split(
    episode_ids: Iterable[int],
    *,
    split_seed: int = 42,
    test_fraction: float = 0.1,
    val_fraction_of_remaining: float = 0.1,
) -> EpisodeSplitManifest:
    """Create an 81/9/10-style split while matching GC-IDM test selection."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    if not 0 < val_fraction_of_remaining < 1:
        raise ValueError("val_fraction_of_remaining must be in (0, 1)")

    unique = np.unique(np.asarray(list(episode_ids), dtype=np.int64))
    if len(unique) < 3:
        raise ValueError("at least three episodes are required for train/val/test")

    rng = np.random.default_rng(split_seed)
    n_test = max(1, round(len(unique) * test_fraction))
    n_test = min(n_test, len(unique) - 2)
    test = np.sort(rng.choice(unique, size=n_test, replace=False))
    remaining = np.setdiff1d(unique, test, assume_unique=True)

    n_val = max(1, round(len(remaining) * val_fraction_of_remaining))
    n_val = min(n_val, len(remaining) - 1)
    val = np.sort(rng.choice(remaining, size=n_val, replace=False))
    train = np.setdiff1d(remaining, val, assume_unique=True)

    manifest = EpisodeSplitManifest(
        split_seed=int(split_seed),
        test_fraction=float(test_fraction),
        val_fraction_of_remaining=float(val_fraction_of_remaining),
        train_episode_ids=train.astype(int).tolist(),
        val_episode_ids=val.astype(int).tolist(),
        test_episode_ids=test.astype(int).tolist(),
    )
    manifest.validate(unique)
    return manifest


def save_episode_split(manifest: EpisodeSplitManifest, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, ensure_ascii=True)
    return output


def load_episode_split(
    path: str | Path,
    expected_episode_ids: Iterable[int] | None = None,
) -> EpisodeSplitManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = EpisodeSplitManifest.from_dict(json.load(handle))
    manifest.validate(expected_episode_ids)
    return manifest


def load_or_create_episode_split(
    path: str | Path,
    episode_ids: Iterable[int],
    *,
    split_seed: int = 42,
    test_fraction: float = 0.1,
    val_fraction_of_remaining: float = 0.1,
) -> EpisodeSplitManifest:
    output = Path(path)
    if output.exists():
        return load_episode_split(output, episode_ids)
    manifest = create_episode_split(
        episode_ids,
        split_seed=split_seed,
        test_fraction=test_fraction,
        val_fraction_of_remaining=val_fraction_of_remaining,
    )
    save_episode_split(manifest, output)
    return manifest
