from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"

from collections import deque
import time
from typing import Any, Dict, Optional

import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from eval import get_dataset, get_episodes_length, img_transform
from latent_subgoal_act.action_priors.common import PixelEncoderMixin, resolve_policy_ckpt
from latent_subgoal_act.action_priors.deterministic_model import GoalConditionedActionPrior
from latent_subgoal_act.wm_rollout import rollout_latent_with_actions


def build_default_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "seed": 42,
            "policy_ckpt": None,
            "lewm_policy": "pusht/lewm",
            "device": "cuda",
            "cache_dir": None,
            "world": {"env_name": "swm/PushT-v1", "num_envs": 50, "max_episode_steps": 100},
            "dataset": {"stats": "pusht_expert_train", "keys_to_cache": ["action", "proprio", "state"]},
            "plan_config": {"receding_horizon": 1},
            "eval": {
                "num_eval": 50,
                "goal_offset_steps": 25,
                "eval_budget": 50,
                "img_size": 224,
                "dataset_name": "pusht_expert_train",
                "callables": [
                    {"method": "_set_state", "args": {"state": {"value": "state"}}},
                    {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
                ],
            },
            "output": {"filename": "goal_action_prior_results.txt"},
            "rerank": {"enabled": True, "num_candidates": 16, "noise_std": 0.2},
            "cem": {"enabled": False, "num_iters": 3, "num_candidates": 64, "elite_frac": 0.1, "init_std": 0.5, "min_std": 0.05},
        }
    )


class GoalActionPriorWorldPolicy(PixelEncoderMixin):
    def __init__(
        self,
        lewm_encoder,
        action_prior,
        transform,
        action_mean,
        action_scale,
        device="cuda",
        execute_horizon=1,
        rerank_enabled=True,
        rerank_num_candidates=16,
        rerank_noise_std=0.2,
        cem_enabled=False,
        cem_num_iters=3,
        cem_num_candidates=64,
        cem_elite_frac=0.1,
        cem_init_std=0.5,
        cem_min_std=0.05,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lewm_encoder = lewm_encoder.to(self.device).eval().requires_grad_(False)
        self.action_prior = action_prior.to(self.device).eval()
        self.transform = transform
        self.action_mean = self._optional_tensor(action_mean)
        self.action_scale = self._optional_tensor(action_scale)
        self.execute_horizon = max(1, int(execute_horizon))
        self.rerank_enabled = bool(rerank_enabled)
        self.rerank_num_candidates = max(1, int(rerank_num_candidates))
        self.rerank_noise_std = float(rerank_noise_std)
        self.cem_enabled = bool(cem_enabled)
        self.cem_num_iters = max(1, int(cem_num_iters))
        self.cem_num_candidates = max(2, int(cem_num_candidates))
        self.cem_elite_frac = float(cem_elite_frac)
        self.cem_init_std = float(cem_init_std)
        self.cem_min_std = float(cem_min_std)
        self._action_buffer = deque()

    def reset(self, *args, **kwargs):
        self._action_buffer.clear()

    def set_env(self, envs):
        self.envs = envs

    def set_envs(self, envs):
        self.set_env(envs)

    def __call__(self, info: Dict[str, Any]):
        return self.act(info)

    @torch.no_grad()
    def act(self, info: Dict[str, Any]):
        if self._action_buffer:
            return self._action_buffer.popleft()
        z_t = self.encode_info_pixels(info, self.PIXEL_KEYS, "pixels")
        z_g = self.encode_info_pixels(info, self.GOAL_KEYS, "goal")
        action = self.action_prior(z_t, z_g)
        if self.cem_enabled:
            action = self._cem(z_t, z_g, action)
        elif self.rerank_enabled:
            action = self._rerank(z_t, z_g, action)
        action = self._inverse_action_scale(action).detach().cpu().numpy()
        self._fill_action_buffer(action)
        return self._action_buffer.popleft()

    def _rerank(self, z_t, z_g, action):
        if self.rerank_num_candidates <= 1:
            return action
        batch_size, horizon, action_dim = action.shape
        candidates = action.unsqueeze(1).expand(batch_size, self.rerank_num_candidates, horizon, action_dim).clone()
        if self.rerank_noise_std > 0:
            candidates[:, 1:] += torch.randn_like(candidates[:, 1:]) * self.rerank_noise_std
        return self._select_best(z_t, z_g, candidates)

    def _cem(self, z_t, z_g, action):
        batch_size, horizon, action_dim = action.shape
        mean = action.detach()
        std = torch.full_like(mean, self.cem_init_std)
        best_candidate = mean
        best_cost = torch.full((batch_size,), float("inf"), device=action.device)
        elite_count = max(1, int(round(self.cem_num_candidates * self.cem_elite_frac)))
        batch_idx = torch.arange(batch_size, device=action.device)
        for _ in range(self.cem_num_iters):
            candidates = mean.unsqueeze(1) + torch.randn(
                batch_size, self.cem_num_candidates, horizon, action_dim, device=action.device, dtype=action.dtype
            ) * std.unsqueeze(1)
            candidates[:, 0] = mean
            cost = self._cost(z_t, z_g, candidates)
            iter_best_cost, iter_best_idx = cost.min(dim=1)
            improved = iter_best_cost < best_cost
            best_candidate = torch.where(improved.view(batch_size, 1, 1), candidates[batch_idx, iter_best_idx], best_candidate)
            best_cost = torch.minimum(best_cost, iter_best_cost)
            elite_idx = torch.topk(cost, k=elite_count, dim=1, largest=False).indices
            gather_idx = elite_idx.view(batch_size, elite_count, 1, 1).expand(-1, -1, horizon, action_dim)
            elite = torch.gather(candidates, dim=1, index=gather_idx)
            mean = elite.mean(dim=1)
            std = elite.std(dim=1, unbiased=False).clamp_min(self.cem_min_std)
        return best_candidate

    def _select_best(self, z_t, z_g, candidates):
        cost = self._cost(z_t, z_g, candidates)
        best = cost.argmin(dim=1)
        batch_idx = torch.arange(candidates.shape[0], device=candidates.device)
        return candidates[batch_idx, best]

    def _cost(self, z_t, z_g, candidates):
        batch_size, num_candidates, horizon, action_dim = candidates.shape
        flat_candidates = candidates.reshape(batch_size * num_candidates, horizon, action_dim)
        flat_z_t = z_t.unsqueeze(1).expand(batch_size, num_candidates, -1).reshape(batch_size * num_candidates, -1)
        rollout = rollout_latent_with_actions(self.lewm_encoder, flat_z_t, flat_candidates)
        rollout = rollout.reshape(batch_size, num_candidates, -1)
        return F.mse_loss(rollout, z_g.unsqueeze(1).expand_as(rollout), reduction="none").mean(dim=-1)

    def _fill_action_buffer(self, action_chunk: np.ndarray):
        horizon = min(self.execute_horizon, action_chunk.shape[1])
        batched = action_chunk.shape[0] > 1
        for step in range(horizon):
            action = action_chunk[:, step, :] if batched else action_chunk[0, step, :]
            self._action_buffer.append(action)

    def _inverse_action_scale(self, action):
        if self.action_mean is None or self.action_scale is None:
            return action
        return action * self.action_scale + self.action_mean

    def _optional_tensor(self, value):
        if value is None:
            return None
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)


def _load_policy(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GoalConditionedActionPrior(**ckpt["model_config"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, ckpt


def run(cfg: DictConfig):
    cfg = OmegaConf.merge(build_default_cfg(), cfg)
    if cfg.policy_ckpt is None:
        raise ValueError("Set policy_ckpt=<experiment>/policy.pt")
    policy_ckpt = resolve_policy_ckpt(cfg.policy_ckpt)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    action_prior, ckpt = _load_policy(policy_ckpt, device)
    lewm = swm.wm.utils.load_pretrained(cfg.lewm_policy).to(device).eval().requires_grad_(False)
    lewm.interpolate_pos_encoding = True
    metadata = ckpt["metadata"]
    policy = GoalActionPriorWorldPolicy(
        lewm_encoder=lewm,
        action_prior=action_prior,
        transform=transform,
        action_mean=metadata.get("action_mean"),
        action_scale=metadata.get("action_scale"),
        device=device,
        execute_horizon=int(cfg.plan_config.receding_horizon),
        rerank_enabled=bool(cfg.rerank.enabled),
        rerank_num_candidates=int(cfg.rerank.num_candidates),
        rerank_noise_std=float(cfg.rerank.noise_std),
        cem_enabled=bool(cfg.cem.enabled),
        cem_num_iters=int(cfg.cem.num_iters),
        cem_num_candidates=int(cfg.cem.num_candidates),
        cem_elite_frac=float(cfg.cem.elite_frac),
        cem_init_std=float(cfg.cem.init_std),
        cem_min_std=float(cfg.cem.min_std),
    )

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    g = np.random.default_rng(cfg.seed)
    random_episode_indices = np.sort(g.choice(valid_indices, size=cfg.eval.num_eval, replace=False))
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
    with (results_path / cfg.output.filename).open("a", encoding="utf-8") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    import sys

    run(OmegaConf.from_cli(sys.argv[1:]))

