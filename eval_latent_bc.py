import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from eval import get_dataset, get_episodes_length, img_transform
from latent_bc import LatentBCWorldPolicy, LatentGoalBCPolicy


def _episode_column(dataset):
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def _load_policy(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LatentGoalBCPolicy(**ckpt["model_config"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, ckpt


def _resolve_model_policy(cfg, metadata):
    policy = cfg.get("lewm_policy", None) or cfg.get("policy", None)
    if policy in (None, "random"):
        policy = metadata.get("model_policy", None)
    if policy in (None, "random"):
        policy = "pusht/lewm"
    return policy


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    policy_ckpt = cfg.get("policy_ckpt")
    if policy_ckpt is None:
        raise ValueError("Set policy_ckpt=experiments/.../policy.pt")
    policy_ckpt = Path(to_absolute_path(policy_ckpt))

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = _episode_column(dataset)
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    bc_policy, ckpt = _load_policy(policy_ckpt, device)
    metadata = ckpt["metadata"]
    model_policy = _resolve_model_policy(cfg, metadata)

    lewm = swm.wm.utils.load_pretrained(model_policy)
    lewm = lewm.to(device).eval()
    lewm.requires_grad_(False)
    lewm.interpolate_pos_encoding = True

    policy = LatentBCWorldPolicy(
        lewm_encoder=lewm,
        bc_policy=bc_policy,
        transform=transform,
        action_mean=metadata.get("action_mean"),
        action_scale=metadata.get("action_scale"),
        device=device,
        execute_horizon=int(cfg.plan_config.receding_horizon),
    )

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]
    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    results_path = policy_ckpt.resolve().parent
    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    end_time = time.time()
    print(metrics)

    output_file = results_path / cfg.output.filename
    with output_file.open("a") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
