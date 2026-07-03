import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing

from eval import get_dataset, get_episodes_length, img_transform


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
def _encode(model, transform, pixels, batch_size, device):
    chunks = []
    model.eval()
    for start in range(0, len(pixels), batch_size):
        batch = _transform_images(transform, pixels[start : start + batch_size])
        batch = batch.unsqueeze(1).to(device)
        output = model.encode({"pixels": batch})
        chunks.append(output["emb"][:, -1].detach().cpu())
    return torch.cat(chunks, dim=0)


def _valid_start_indices(dataset, goal_offset, action_horizon):
    col_name = _episode_column(dataset)
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    episodes, _ = np.unique(episode_idx, return_index=True)
    lengths = get_episodes_length(dataset, episodes)
    max_start = lengths - max(goal_offset, action_horizon) - 1
    max_start_by_episode = {
        ep_id: max_start[i] for i, ep_id in enumerate(episodes) if max_start[i] >= 0
    }
    valid_mask = np.array(
        [
            ep in max_start_by_episode and step <= max_start_by_episode[ep]
            for ep, step in zip(episode_idx, step_idx)
        ]
    )
    return np.nonzero(valid_mask)[0]


def _action_chunks(dataset, indices, action_horizon, scaler):
    actions = dataset.get_col_data("action")
    raw = np.stack(
        [actions[idx : idx + action_horizon] for idx in indices],
        axis=0,
    ).astype("float32")
    flat = raw.reshape(-1, raw.shape[-1])
    normalized = scaler.transform(flat).reshape(raw.shape).astype("float32")
    return raw, normalized


def _resolve_model_policy(cfg):
    policy = cfg.get("lewm_policy", None) or cfg.get("policy", None)
    if policy in (None, "random"):
        policy = "pusht/lewm"
    return policy


@hydra.main(version_base=None, config_path="../config/eval", config_name="pusht")
def run(cfg: DictConfig):
    output = Path(
        to_absolute_path(
            cfg.get("output_dataset", "experiments/latent_bc_datasets/pusht_g25_k5.pt")
        )
    )
    max_samples = cfg.get("max_samples", None)
    batch_size = int(cfg.get("encode_batch_size", 128))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model_policy = _resolve_model_policy(cfg)

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    transform = img_transform(cfg)
    action_scaler = _fit_action_scaler(dataset)

    goal_offset = int(cfg.eval.goal_offset_steps)
    action_horizon = int(cfg.plan_config.action_block)
    valid_indices = _valid_start_indices(dataset, goal_offset, action_horizon)
    if max_samples is not None:
        valid_indices = valid_indices[: int(max_samples)]
    if len(valid_indices) == 0:
        raise ValueError("No valid samples found for latent BC dataset.")

    model = swm.wm.utils.load_pretrained(model_policy)
    model = model.to(device)
    model.eval().requires_grad_(False)
    model.interpolate_pos_encoding = True

    rows = dataset.get_row_data(valid_indices)
    goal_rows = dataset.get_row_data(valid_indices + goal_offset)
    col_name = _episode_column(dataset)

    z_t = _encode(model, transform, rows["pixels"], batch_size, device)
    z_g = _encode(model, transform, goal_rows["pixels"], batch_size, device)
    action_raw, action = _action_chunks(
        dataset, valid_indices, action_horizon, action_scaler
    )

    payload = {
        "z_t": z_t.float(),
        "z_g": z_g.float(),
        "delta_z": (z_g - z_t).float(),
        "action": torch.from_numpy(action),
        "action_raw": torch.from_numpy(action_raw),
        "episode": torch.as_tensor(rows[col_name], dtype=torch.long),
        "step": torch.as_tensor(rows["step_idx"], dtype=torch.long),
        "goal_step": torch.as_tensor(goal_rows["step_idx"], dtype=torch.long),
        "metadata": {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "dataset_name": cfg.eval.dataset_name,
            "model_policy": model_policy,
            "goal_offset_steps": goal_offset,
            "action_horizon": action_horizon,
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
    run()
