from __future__ import annotations

from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from eval import get_dataset, get_episodes_length, img_transform
from latent_subgoal_act.shared import datasets_root


def build_default_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "output_dataset": "pusht_g25_k5_h5.pt",
            "max_samples": None,
            "sample_mode": "fixed_offset",
            "goal_stride": 25,
            "split": "train",
            "split_seed": 42,
            "test_fraction": 0.1,
            "encode_batch_size": 128,
            "device": "cuda",
            "lewm_policy": "pusht/lewm",
            "cache_dir": None,
            "dataset": {"keys_to_cache": ["action", "proprio", "state"]},
            "eval": {
                "dataset_name": "pusht_expert_train",
                "goal_offset_steps": 25,
                "img_size": 224,
            },
            "plan_config": {
                "action_block": 5,
                "subgoal_horizon": 5,
                "cap_subgoal_at_goal": True,
            },
        }
    )


def _episode_column(dataset):
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def _fit_action_scaler(dataset):
    scaler = preprocessing.StandardScaler()
    action = dataset.get_col_data("action")
    action = action[~np.isnan(action).any(axis=1)]
    scaler.fit(action)
    return scaler


def _transform_images(transform, images):
    return torch.stack([transform(img) for img in images], dim=0)


@torch.no_grad()
def _encode_indices(model, dataset, transform, indices, batch_size, device, desc):
    indices = np.asarray(indices, dtype=np.int64)
    unique_indices, inverse = np.unique(indices, return_inverse=True)
    chunks = []
    model.eval()
    progress = tqdm(total=len(unique_indices), desc=desc, unit="sample") if tqdm is not None else None
    for start in range(0, len(unique_indices), batch_size):
        batch_indices = unique_indices[start : start + batch_size]
        rows = dataset.get_row_data(batch_indices)
        batch = _transform_images(transform, rows["pixels"])
        batch = batch.unsqueeze(1).to(device)
        output = model.encode({"pixels": batch})
        chunks.append(output["emb"][:, -1].detach().cpu())
        if progress is not None:
            progress.update(len(batch_indices))
    if progress is not None:
        progress.close()
    encoded_unique = torch.cat(chunks, dim=0)
    return encoded_unique[torch.as_tensor(inverse, dtype=torch.long)]


def _episode_split_ids(dataset, seed, test_fraction):
    col_name = _episode_column(dataset)
    episode_idx = dataset.get_col_data(col_name)
    episodes = np.unique(episode_idx)
    rng = np.random.default_rng(seed)
    episodes = np.array(episodes)
    rng.shuffle(episodes)
    n_test = max(1, int(round(len(episodes) * float(test_fraction))))
    test_eps = set(episodes[:n_test].tolist())
    train_eps = set(episodes[n_test:].tolist())
    return train_eps, test_eps


def _fixed_offset_indices(dataset, goal_offset, action_horizon, subgoal_horizon, cap_subgoal_at_goal, episode_subset=None):
    col_name = _episode_column(dataset)
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    episodes, _ = np.unique(episode_idx, return_index=True)
    lengths = get_episodes_length(dataset, episodes)
    needed = max(goal_offset, action_horizon, subgoal_horizon)
    max_start = lengths - needed - 1
    max_start_by_episode = {
        ep_id: max_start[i] for i, ep_id in enumerate(episodes) if max_start[i] >= 0
    }
    valid_mask = np.array(
        [
            ep in max_start_by_episode
            and (episode_subset is None or ep in episode_subset)
            and step <= max_start_by_episode[ep]
            for ep, step in zip(episode_idx, step_idx)
        ]
    )
    start_indices = np.nonzero(valid_mask)[0]
    goal_indices = start_indices + goal_offset
    if cap_subgoal_at_goal:
        subgoal_indices = np.minimum(start_indices + subgoal_horizon, goal_indices)
    else:
        subgoal_indices = start_indices + subgoal_horizon
    return start_indices, goal_indices, subgoal_indices


def _future_subgoal_indices(start_indices, goal_indices, subgoal_horizon, cap_subgoal_at_goal):
    offsets = np.arange(1, int(subgoal_horizon) + 1, dtype=np.int64)
    future = start_indices[:, None] + offsets[None, :]
    if cap_subgoal_at_goal:
        future = np.minimum(future, goal_indices[:, None])
    return future


def _goal_anchored_indices(dataset, goal_offset, action_horizon, subgoal_horizon, goal_stride, cap_subgoal_at_goal, episode_subset=None):
    col_name = _episode_column(dataset)
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    starts, goals, subgoals = [], [], []
    progress = tqdm(desc="building goal-anchored index", unit="episode") if tqdm is not None else None

    for ep in np.unique(episode_idx):
        if episode_subset is not None and ep not in episode_subset:
            continue
        rows = np.nonzero(episode_idx == ep)[0]
        rows = rows[np.argsort(step_idx[rows])]
        if len(rows) <= max(action_horizon, subgoal_horizon):
            continue

        first_goal_pos = max(1, min(goal_offset, len(rows) - 1))
        stride = max(1, int(goal_stride))
        for goal_pos in range(first_goal_pos, len(rows), stride):
            start_min = max(0, goal_pos - goal_offset)
            for start_pos in range(start_min, goal_pos):
                if start_pos + action_horizon >= len(rows):
                    continue
                subgoal_pos = start_pos + subgoal_horizon
                if cap_subgoal_at_goal:
                    subgoal_pos = min(subgoal_pos, goal_pos)
                elif subgoal_pos >= len(rows):
                    continue
                starts.append(rows[start_pos])
                goals.append(rows[goal_pos])
                subgoals.append(rows[subgoal_pos])

        if progress is not None:
            progress.update(1)

    if progress is not None:
        progress.close()

    return np.asarray(starts, dtype=np.int64), np.asarray(goals, dtype=np.int64), np.asarray(subgoals, dtype=np.int64)


def _action_chunks(dataset, indices, action_horizon, scaler):
    actions = dataset.get_col_data("action")
    raw = np.stack([actions[idx : idx + action_horizon] for idx in indices], axis=0).astype("float32")
    flat = raw.reshape(-1, raw.shape[-1])
    normalized = scaler.transform(flat).reshape(raw.shape).astype("float32")
    return raw, normalized


def run(cfg: DictConfig):
    cfg = OmegaConf.merge(build_default_cfg(), cfg)
    output = Path(cfg.output_dataset).expanduser()
    if not output.is_absolute():
        output = datasets_root() / output

    device = cfg.device if torch.cuda.is_available() else "cpu"
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    transform = img_transform(cfg)
    action_scaler = _fit_action_scaler(dataset)

    goal_offset = int(cfg.eval.goal_offset_steps)
    action_horizon = int(cfg.plan_config.action_block)
    subgoal_horizon = int(cfg.plan_config.subgoal_horizon)
    train_eps, test_eps = _episode_split_ids(dataset, cfg.split_seed, cfg.test_fraction)
    if cfg.split == "train":
        episode_subset = train_eps
    elif cfg.split in {"test", "eval", "val"}:
        episode_subset = test_eps
    elif cfg.split == "all":
        episode_subset = None
    else:
        raise ValueError("split must be one of: train, test, eval, val, all")

    if cfg.sample_mode == "fixed_offset":
        valid_indices, goal_indices, subgoal_indices = _fixed_offset_indices(
            dataset,
            goal_offset,
            action_horizon,
            subgoal_horizon,
            bool(cfg.plan_config.cap_subgoal_at_goal),
            episode_subset=episode_subset,
        )
    elif cfg.sample_mode == "goal_anchored":
        valid_indices, goal_indices, subgoal_indices = _goal_anchored_indices(
            dataset,
            goal_offset,
            action_horizon,
            subgoal_horizon,
            int(cfg.goal_stride),
            bool(cfg.plan_config.cap_subgoal_at_goal),
            episode_subset=episode_subset,
        )
    else:
        raise ValueError("sample_mode must be 'fixed_offset' or 'goal_anchored'")

    if cfg.max_samples is not None:
        keep = slice(0, int(cfg.max_samples))
        valid_indices = valid_indices[keep]
        goal_indices = goal_indices[keep]
        subgoal_indices = subgoal_indices[keep]
    if len(valid_indices) == 0:
        raise ValueError("No valid samples found for latent subgoal ACT dataset.")

    model = swm.wm.utils.load_pretrained(cfg.lewm_policy)
    model = model.to(device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True

    subgoal_sequence_indices = _future_subgoal_indices(
        valid_indices,
        goal_indices,
        subgoal_horizon,
        bool(cfg.plan_config.cap_subgoal_at_goal),
    )

    z_t = _encode_indices(model, dataset, transform, valid_indices, int(cfg.encode_batch_size), device, "encoding z_t")
    z_h_seq_flat = _encode_indices(
        model,
        dataset,
        transform,
        subgoal_sequence_indices.reshape(-1),
        int(cfg.encode_batch_size),
        device,
        "encoding z_h_seq",
    )
    z_h_seq = z_h_seq_flat.reshape(len(valid_indices), subgoal_horizon, -1)
    z_h = z_h_seq[:, -1]
    z_g = _encode_indices(model, dataset, transform, goal_indices, int(cfg.encode_batch_size), device, "encoding z_g")

    action_raw, action = _action_chunks(dataset, valid_indices, action_horizon, action_scaler)
    col_name = _episode_column(dataset)
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")

    payload = {
        "z_t": z_t.float(),
        "z_g": z_g.float(),
        "z_h": z_h.float(),
        "z_h_seq": z_h_seq.float(),
        "action": torch.from_numpy(action),
        "action_raw": torch.from_numpy(action_raw),
        "episode": torch.as_tensor(episode_idx[valid_indices], dtype=torch.long),
        "step": torch.as_tensor(step_idx[valid_indices], dtype=torch.long),
        "subgoal_step": torch.as_tensor(step_idx[subgoal_indices], dtype=torch.long),
        "subgoal_steps": torch.as_tensor(step_idx[subgoal_sequence_indices], dtype=torch.long),
        "goal_step": torch.as_tensor(step_idx[goal_indices], dtype=torch.long),
        "metadata": {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "dataset_name": cfg.eval.dataset_name,
            "model_policy": cfg.lewm_policy,
            "sample_mode": cfg.sample_mode,
            "goal_stride": int(cfg.goal_stride),
            "goal_offset_steps": goal_offset,
            "subgoal_horizon": subgoal_horizon,
            "cap_subgoal_at_goal": bool(cfg.plan_config.cap_subgoal_at_goal),
            "action_horizon": action_horizon,
            "split": cfg.split,
            "split_seed": int(cfg.split_seed),
            "test_fraction": float(cfg.test_fraction),
            "num_train_episodes": int(len(train_eps)),
            "num_test_episodes": int(len(test_eps)),
            "action_dim": int(action.shape[-1]),
            "latent_dim": int(z_t.shape[-1]),
            "action_mean": action_scaler.mean_.astype("float32"),
            "action_scale": action_scaler.scale_.astype("float32"),
            "num_samples": int(len(valid_indices)),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"saved {len(valid_indices)} samples to {output}")


if __name__ == "__main__":
    import sys

    cli = OmegaConf.from_cli(sys.argv[1:])
    run(cli)
