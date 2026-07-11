#!/usr/bin/env python3
"""Closed-loop environment evaluation for a single-step token IDM.

This follows the GC-IDM evaluation protocol: select fixed dataset goals, encode
the current observation online at every step, cache the goal encoding, decode
one action, execute it, and report environment success metrics.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from dataclasses import dataclass

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import stable_pretraining as spt
import stable_worldmodel as swm
from sklearn import preprocessing
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from single_step_token_idm.dataset import load_lewm_model
from single_step_token_idm.model import GoalConditionedTokenIDM, TokenIDMConfig
from single_step_token_idm.tokenization import ActionTokenizer


@dataclass
class EvalConfig:
    env: str
    dataset_name: str
    cache_keys: list[str]
    img_size: int
    callables: list[dict]
    world_kwargs: dict | None = None


DATASET_CONFIGS = {
    "pusht": EvalConfig(
        env="swm/PushT-v1",
        dataset_name="pusht_expert_train",
        cache_keys=["action", "proprio", "state"],
        img_size=224,
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
        ],
    ),
    "tworoom": EvalConfig(
        env="swm/TwoRoom-v1",
        dataset_name="tworoom",
        cache_keys=["action", "proprio"],
        img_size=224,
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "proprio"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_proprio"}}},
        ],
    ),
    "cube": EvalConfig(
        env="swm/OGBCube-v0",
        dataset_name="ogbench/cube_single_expert",
        cache_keys=["action"],
        img_size=224,
        callables=[
            {"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
            {"method": "set_target_pos", "args": {
                "cube_id": {"value": 0, "in_dataset": False},
                "target_pos": {"value": "goal_privileged_block_0_pos"},
                "target_quat": {"value": "goal_privileged_block_0_quat"},
            }},
        ],
        world_kwargs={"env_type": "single", "ob_type": "states", "multiview": False,
                      "width": 224, "height": 224, "visualize_info": False,
                      "terminate_at_goal": True},
    ),
    "reacher": EvalConfig(
        env="swm/ReacherDMControl-v0",
        dataset_name="dmc/reacher_random",
        cache_keys=["action"],
        img_size=224,
        callables=[
            {"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
            {"method": "set_target_qpos", "args": {"target_qpos": {"value": "goal_qpos"}}},
        ],
        world_kwargs={"task": "qpos_match"},
    ),
}


def get_img_transform(img_size: int):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


def build_normalizers(dataset, keys: list[str]) -> dict:
    process = {}
    for col in keys:
        if col == "pixels":
            continue
        values = dataset.get_col_data(col)
        values = values[~np.isnan(values).any(axis=1)]
        scaler = preprocessing.StandardScaler().fit(values)
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler
    return process


def sample_eval_episodes(dataset, num_eval: int, goal_offset: int, seed: int,
                         train_split: float, split_seed: int):
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids = dataset.get_col_data(col)
    step_ids = dataset.get_col_data("step_idx")
    unique_eps = np.unique(episode_ids)
    lengths = {ep: np.max(step_ids[episode_ids == ep]) + 1 for ep in unique_eps}
    valid = np.array([step <= lengths[ep] - goal_offset - 1
                      for ep, step in zip(episode_ids, step_ids)])

    if train_split < 1.0:
        n_holdout = max(1, round(len(unique_eps) * (1.0 - train_split)))
        split_rng = np.random.default_rng(split_seed)
        holdout = split_rng.choice(unique_eps, size=n_holdout, replace=False)
        valid &= np.isin(episode_ids, holdout)
        print(f"Evaluation uses {len(holdout)} held-out episodes: {sorted(holdout.tolist())}")

    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        raise RuntimeError("No valid evaluation starts for this goal offset and split")
    n = min(num_eval, len(valid_indices))
    rng = np.random.default_rng(seed)
    sampled = np.sort(rng.choice(valid_indices, size=n, replace=False))
    rows = dataset.get_row_data(sampled)
    return rows[col], rows["step_idx"]


def load_token_idm(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = TokenIDMConfig(**checkpoint["config"])
    model = GoalConditionedTokenIDM(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = ActionTokenizer.from_dict(checkpoint["tokenizer"])
    return model, tokenizer, checkpoint


class TokenIDMPolicy:
    """Online LeWM encoding followed by one token-IDM action per step."""

    def __init__(self, jepa, idm, tokenizer, eval_budget: int, process: dict,
                 transform: dict, device: torch.device, cache_goal: bool,
                 do_sample: bool, temperature: float):
        self.jepa = jepa
        self.idm = idm
        self.tokenizer = tokenizer
        self.eval_budget = eval_budget
        self.process = process
        self.transform = transform
        self.device = device
        self.cache_goal = cache_goal
        self.do_sample = do_sample
        self.temperature = temperature
        self._step_count = 0
        self._cached_goal_input = None
        self._cached_z_goal = None
        self._plan_times: list[float] = []

    def set_env(self, env):
        self.env = env

    def _prepare_info(self, info: dict) -> dict:
        out = {}
        for key, value in info.items():
            is_numpy = isinstance(value, (np.ndarray, np.generic))
            if key in self.process and is_numpy:
                shape = value.shape
                flat = value.reshape(-1, *shape[2:]) if len(shape) > 2 else value
                value = self.process[key].transform(flat).reshape(shape)
            if key in self.transform:
                shape = value.shape if value.ndim > 2 else None
                if shape is not None:
                    value = value.reshape(-1, *shape[2:])
                value = np.transpose(value, (0, 3, 1, 2)) if isinstance(value, np.ndarray) else value.permute(0, 3, 1, 2)
                value = torch.stack([self.transform[key](tv_tensors.Image(x)) for x in value])
                if shape is not None:
                    value = value.reshape(*shape[:2], *value.shape[1:])
                is_numpy = False
            if is_numpy and value.dtype.kind not in "USO":
                value = torch.from_numpy(value)
            out[key] = value
        return out

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        encoded = self.jepa.encoder(pixels, interpolate_pos_encoding=True)
        return self.jepa.projector(encoded.last_hidden_state[:, 0])

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        start = time.perf_counter()
        info = self._prepare_info(info_dict)
        pixels = info["pixels"][:, -1].to(self.device)
        goal = info["goal"][:, -1].to(self.device)
        z_current = self._encode(pixels)

        cache_hit = (self.cache_goal and self._cached_goal_input is not None
                     and self._cached_goal_input.shape == goal.shape
                     and torch.equal(self._cached_goal_input, goal))
        if cache_hit:
            z_goal = self._cached_z_goal
        else:
            z_goal = self._encode(goal)
            if self.cache_goal:
                self._cached_goal_input = goal.detach().clone()
                self._cached_z_goal = z_goal

        remaining = max(1, min(self.eval_budget - self._step_count, self.idm.max_horizon))
        steps = torch.full((z_current.shape[0],), remaining, device=self.device, dtype=torch.long)
        logits = self.idm(z_current, z_goal, steps)
        bin_ids = self.idm.predict_token_ids(
            logits, do_sample=self.do_sample, temperature=self.temperature
        )
        token_ids = bin_ids.cpu().numpy() + self.tokenizer.token_offset
        actions = self.tokenizer.token_ids_to_actions(token_ids)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._plan_times.append(time.perf_counter() - start)
        self._step_count += 1
        return actions.reshape(*self.env.action_space.shape)

    @property
    def avg_plan_time_ms(self) -> float:
        return 0.0 if not self._plan_times else 1000 * sum(self._plan_times) / len(self._plan_times)


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop token IDM evaluation")
    parser.add_argument("--dataset", required=True, choices=DATASET_CONFIGS)
    parser.add_argument("--idm", required=True, help="Token IDM checkpoint")
    parser.add_argument("--checkpoint", default=None, help="LeWM checkpoint or directory")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--dataset-name-override", default=None)
    parser.add_argument("--no-goal-cache", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.num_eval < 1 or args.goal_offset < 1 or args.eval_budget < 1:
        parser.error("--num-eval, --goal-offset, and --eval-budget must be positive")

    device = torch.device(args.device)
    idm, tokenizer, saved = load_token_idm(args.idm, device)
    train_split = args.train_split if args.train_split is not None else saved.get("train_split", 1.0)
    split_seed = args.split_seed if args.split_seed is not None else saved.get("split_seed", 42)
    if not 0 < train_split <= 1:
        parser.error("--train-split must be in (0, 1]")

    dcfg = DATASET_CONFIGS[args.dataset]
    if args.dataset_name_override:
        dcfg = dataclasses.replace(dcfg, dataset_name=args.dataset_name_override)
    data_dir = os.environ.get("STABLEWM_HOME", "./stable-wm-data")
    lewm_path = args.checkpoint or os.path.join(data_dir, "checkpoints", args.dataset, "lewm")

    dataset = swm.data.HDF5Dataset(dcfg.dataset_name, keys_to_cache=dcfg.cache_keys)
    transform = {"pixels": get_img_transform(dcfg.img_size), "goal": get_img_transform(dcfg.img_size)}
    process = build_normalizers(dataset, dcfg.cache_keys)
    episodes, starts = sample_eval_episodes(
        dataset, args.num_eval, args.goal_offset, args.seed, train_split, split_seed
    )

    lewm = load_lewm_model(lewm_path, str(device))
    jepa = lewm.model if hasattr(lewm, "model") else lewm
    jepa.eval().requires_grad_(False)
    policy = TokenIDMPolicy(
        jepa, idm, tokenizer, args.eval_budget, process, transform, device,
        cache_goal=not args.no_goal_cache, do_sample=args.sample,
        temperature=args.temperature,
    )
    world = swm.World(
        env_name=dcfg.env,
        num_envs=len(episodes),
        image_shape=(dcfg.img_size, dcfg.img_size),
        max_episode_steps=2 * max(args.eval_budget, args.goal_offset) + 5,
        **(dcfg.world_kwargs or {}),
    )
    world.set_policy(policy)

    started = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=starts.tolist(),
        goal_offset=args.goal_offset,
        eval_budget=args.eval_budget,
        episodes_idx=episodes.tolist(),
        callables=dcfg.callables,
    )
    elapsed = time.time() - started
    print(f"Token IDM metrics: {metrics}")
    print(f"Wall-clock: {elapsed:.1f}s total, {1000 * elapsed / len(episodes):.0f} ms/episode")
    print(f"Average policy time: {policy.avg_plan_time_ms:.2f} ms/action")


if __name__ == "__main__":
    main()
