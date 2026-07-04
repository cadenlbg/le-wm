from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"

import time

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

from eval import get_dataset, get_episodes_length, img_transform
from latent_subgoal_act.model import LatentSubgoalACTPolicy
from latent_subgoal_act.policy import LatentSubgoalACTWorldPolicy
from latent_subgoal_act.shared import resolve_experiment_path


def build_default_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "seed": 42,
            "policy": "random",
            "policy_ckpt": None,
            "lewm_policy": "pusht/lewm",
            "device": "cuda",
            "cache_dir": None,
            "world": {
                "env_name": "swm/PushT-v1",
                "num_envs": 50,
                "max_episode_steps": 100,
            },
            "dataset": {
                "stats": "pusht_expert_train",
                "keys_to_cache": ["action", "proprio", "state"],
            },
            "plan_config": {
                "horizon": 5,
                "receding_horizon": 1,
                "action_block": 5,
            },
            "eval": {
                "num_eval": 50,
                "goal_offset_steps": 25,
                "eval_budget": 50,
                "img_size": 224,
                "dataset_name": "pusht_expert_train",
                "split": "test",
                "split_seed": 42,
                "test_fraction": 0.1,
                "callables": [
                    {"method": "_set_state", "args": {"state": {"value": "state"}}},
                    {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
                ],
            },
            "output": {"filename": None},
            "rerank": {
                "enabled": True,
                "num_candidates": 16,
                "noise_std": 0.2,
                "target": "subgoal",
            },
            "cem": {
                "enabled": False,
                "num_iters": 3,
                "num_candidates": 64,
                "elite_frac": 0.1,
                "init_std": 0.5,
                "min_std": 0.05,
            },
            "temporal_ensemble": {
                "enabled": False,
                "decay": 0.01,
            },
        }
    )


def _episode_column(dataset):
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def _episode_split_ids(dataset, seed, test_fraction):
    col_name = _episode_column(dataset)
    episodes = np.unique(dataset.get_col_data(col_name))
    episodes = np.array(episodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_test = max(1, int(round(len(episodes) * float(test_fraction))))
    return set(episodes[n_test:].tolist()), set(episodes[:n_test].tolist())


def _load_policy(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LatentSubgoalACTPolicy(**ckpt["model_config"])
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


def _default_output_filename(cfg):
    if cfg.output.filename:
        return cfg.output.filename
    if cfg.cem.enabled:
        target = cfg.rerank.target
        return f"pusht_cem_to_{target}_results.txt"
    if cfg.rerank.enabled:
        target = cfg.rerank.target
        return f"pusht_rerank_to_{target}_results.txt"
    return "pusht_direct_results.txt"


def run(cfg: DictConfig):
    cfg = OmegaConf.merge(build_default_cfg(), cfg)
    if cfg.policy_ckpt is None:
        raise ValueError("Set policy_ckpt=<experiment>/policy.pt")
    policy_ckpt = resolve_experiment_path(cfg.policy_ckpt)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = _episode_column(dataset)
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    act_policy, ckpt = _load_policy(policy_ckpt, device)
    metadata = ckpt["metadata"]
    lewm = swm.wm.utils.load_pretrained(_resolve_model_policy(cfg, metadata))
    lewm = lewm.to(device).eval()
    lewm.requires_grad_(False)
    lewm.interpolate_pos_encoding = True

    policy = LatentSubgoalACTWorldPolicy(
        lewm_encoder=lewm,
        act_policy=act_policy,
        transform=transform,
        action_mean=metadata.get("action_mean"),
        action_scale=metadata.get("action_scale"),
        device=device,
        execute_horizon=int(cfg.plan_config.receding_horizon),
        rerank_wm=lewm if (cfg.rerank.enabled or cfg.cem.enabled) else None,
        rerank_num_candidates=int(cfg.rerank.num_candidates),
        rerank_noise_std=float(cfg.rerank.noise_std),
        rerank_target=cfg.rerank.target,
        cem_enabled=bool(cfg.cem.enabled),
        cem_num_iters=int(cfg.cem.num_iters),
        cem_num_candidates=int(cfg.cem.num_candidates),
        cem_elite_frac=float(cfg.cem.elite_frac),
        cem_init_std=float(cfg.cem.init_std),
        cem_min_std=float(cfg.cem.min_std),
        temporal_ensemble=bool(cfg.temporal_ensemble.enabled),
        ensemble_decay=float(cfg.temporal_ensemble.decay),
    )

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)])
    train_eps, test_eps = _episode_split_ids(dataset, cfg.eval.split_seed, cfg.eval.test_fraction)
    if cfg.eval.split == "train":
        eval_eps = train_eps
    elif cfg.eval.split in {"test", "eval", "val"}:
        eval_eps = test_eps
    elif cfg.eval.split == "all":
        eval_eps = None
    else:
        raise ValueError("eval.split must be one of: train, test, eval, val, all")

    episode_per_row = dataset.get_col_data(col_name)
    split_mask = np.ones_like(episode_per_row, dtype=bool) if eval_eps is None else np.array([ep in eval_eps for ep in episode_per_row])
    valid_mask = (dataset.get_col_data("step_idx") <= max_start_per_row) & split_mask
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), f"valid starting points found for {cfg.eval.split} evaluation.")

    g = np.random.default_rng(cfg.seed)
    if len(valid_indices) < cfg.eval.num_eval:
        raise ValueError(
            f"Not enough valid starting points for evaluation: "
            f"need {cfg.eval.num_eval}, found {len(valid_indices)}."
        )
    random_episode_indices = g.choice(valid_indices, size=cfg.eval.num_eval, replace=False)
    random_episode_indices = np.sort(random_episode_indices)
    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]
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

    output_file = results_path / _default_output_filename(cfg)
    with output_file.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    import sys

    run(OmegaConf.from_cli(sys.argv[1:]))
