from __future__ import annotations

import json

import numpy as np
import torch
from omegaconf import OmegaConf

from latent_subgoal_act.shared import resolve_dataset_path


def _episode_split_from_payload(payload, split_seed, test_fraction):
    episodes = torch.unique(payload["episode"].cpu()).numpy()
    episodes = np.array(episodes)
    rng = np.random.default_rng(split_seed)
    rng.shuffle(episodes)
    n_test = max(1, int(round(len(episodes) * float(test_fraction))))
    return set(episodes[n_test:].tolist()), set(episodes[:n_test].tolist())


def run(cfg):
    if cfg.get("dataset") is None:
        raise ValueError("Set dataset=<path-or-name>.pt")

    path = resolve_dataset_path(cfg.dataset)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    required = ["z_t", "z_g", "z_h", "action", "episode", "step", "subgoal_step", "goal_step"]
    missing = [key for key in required if key not in payload]

    episode = payload["episode"].cpu()
    unique_eps = torch.unique(episode)
    split_seed = int(cfg.get("split_seed", metadata.get("split_seed", 42)))
    test_fraction = float(cfg.get("test_fraction", metadata.get("test_fraction", 0.1)))
    train_eps, test_eps = _episode_split_from_payload(payload, split_seed, test_fraction)
    present_eps = set(unique_eps.numpy().tolist())

    expected_split = cfg.get("expected_split", metadata.get("split", None))
    split_ok = None
    if expected_split == "train":
        split_ok = len(present_eps & test_eps) == 0
    elif expected_split in {"test", "eval", "val"}:
        split_ok = len(present_eps & train_eps) == 0

    summary = {
        "path": str(path),
        "num_samples": int(payload["z_t"].shape[0]),
        "num_episodes": int(len(unique_eps)),
        "missing_keys": missing,
        "z_t_shape": list(payload["z_t"].shape),
        "z_g_shape": list(payload["z_g"].shape),
        "z_h_shape": list(payload["z_h"].shape),
        "z_h_seq_shape": list(payload["z_h_seq"].shape) if "z_h_seq" in payload else None,
        "action_shape": list(payload["action"].shape),
        "metadata": {
            "dataset_name": metadata.get("dataset_name"),
            "sample_mode": metadata.get("sample_mode"),
            "goal_stride": metadata.get("goal_stride"),
            "split": metadata.get("split"),
            "split_seed": metadata.get("split_seed"),
            "test_fraction": metadata.get("test_fraction"),
            "goal_offset_steps": metadata.get("goal_offset_steps"),
            "subgoal_horizon": metadata.get("subgoal_horizon"),
            "cap_subgoal_at_goal": metadata.get("cap_subgoal_at_goal"),
            "action_horizon": metadata.get("action_horizon"),
            "model_policy": metadata.get("model_policy"),
        },
        "expected_split": expected_split,
        "split_ok": split_ok,
        "train_episode_overlap": int(len(present_eps & train_eps)),
        "test_episode_overlap": int(len(present_eps & test_eps)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if missing:
        raise SystemExit(2)
    if split_ok is False:
        raise SystemExit(3)


if __name__ == "__main__":
    import sys

    run(OmegaConf.from_cli(sys.argv[1:]))
