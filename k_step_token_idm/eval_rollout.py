#!/usr/bin/env python3
"""Real-environment closed-loop evaluation for K-step token IDM."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from sklearn import preprocessing
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from k_step_token_idm.eval_offline import load_model_checkpoint
from k_step_token_idm.splits import EpisodeSplitManifest
from single_step_token_idm.dataset import load_lewm_model


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
    import stable_pretraining as spt

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def build_normalizers(dataset, keys: list[str]) -> dict:
    process = {}
    for column in keys:
        if column == "pixels":
            continue
        values = dataset.get_col_data(column)
        values = values[~np.isnan(values).any(axis=1)]
        scaler = preprocessing.StandardScaler().fit(values)
        process[column] = scaler
        if column != "action":
            process[f"goal_{column}"] = scaler
    return process


def sample_eval_starts(
    dataset,
    allowed_episode_ids: list[int],
    *,
    num_eval: int,
    goal_offset: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids = dataset.get_col_data(column).astype(np.int64)
    step_ids = dataset.get_col_data("step_idx").astype(np.int64)
    allowed = np.asarray(allowed_episode_ids, dtype=np.int64)
    missing = sorted(set(allowed.tolist()) - set(np.unique(episode_ids).tolist()))
    if missing:
        raise ValueError(f"split episodes are absent from environment dataset: {missing}")

    lengths = {
        int(episode): int(np.max(step_ids[episode_ids == episode]) + 1)
        for episode in np.unique(episode_ids)
    }
    valid = np.isin(episode_ids, allowed)
    valid &= np.asarray(
        [step <= lengths[int(episode)] - goal_offset - 1 for episode, step in zip(episode_ids, step_ids)]
    )
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        raise RuntimeError("no valid closed-loop starts for the selected split and goal offset")
    count = min(num_eval, len(valid_indices))
    rng = np.random.default_rng(seed)
    sampled = np.sort(rng.choice(valid_indices, size=count, replace=False))
    rows = dataset.get_row_data(sampled)
    return rows[column].astype(np.int64), rows["step_idx"].astype(np.int64)


class KStepTokenIDMPolicy:
    """Encode real observations, generate a trunk, and execute a short prefix."""

    def __init__(
        self,
        jepa,
        idm,
        tokenizer,
        *,
        goal_offset: int,
        execute_horizon: int,
        process: dict,
        transform: dict,
        device: torch.device,
        cache_goal: bool,
        do_sample: bool,
        temperature: float,
        top_k: int | None,
    ):
        if not 1 <= execute_horizon <= idm.action_horizon:
            raise ValueError("execute_horizon must be in [1, action_horizon]")
        self.jepa = jepa
        self.idm = idm
        self.tokenizer = tokenizer
        self.goal_offset = int(goal_offset)
        self.execute_horizon = int(execute_horizon)
        self.process = process
        self.transform = transform
        self.device = device
        self.cache_goal = cache_goal
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.env = None
        self._action_buffer = deque()
        self._cached_goal_input = None
        self._cached_z_goal = None
        self._step_count = 0
        self._plan_times: list[float] = []
        self._action_call_times: list[float] = []

    def set_env(self, env):
        self.env = env

    def set_envs(self, env):
        self.set_env(env)

    def reset(self, *args, **kwargs):
        self._action_buffer.clear()
        self._cached_goal_input = None
        self._cached_z_goal = None
        self._step_count = 0

    def _prepare_info(self, info: dict) -> dict:
        output = {}
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
                value = (
                    np.transpose(value, (0, 3, 1, 2))
                    if isinstance(value, np.ndarray)
                    else value.permute(0, 3, 1, 2)
                )
                value = torch.stack(
                    [self.transform[key](tv_tensors.Image(frame)) for frame in value]
                )
                if shape is not None:
                    value = value.reshape(*shape[:2], *value.shape[1:])
                is_numpy = False
            if is_numpy and value.dtype.kind not in "USO":
                value = torch.from_numpy(value)
            output[key] = value
        return output

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        encoded = self.jepa.encoder(pixels, interpolate_pos_encoding=True)
        return self.jepa.projector(encoded.last_hidden_state[:, 0])

    def _reshape_for_env(self, actions: np.ndarray) -> np.ndarray:
        if self.env is None:
            return actions
        return actions.reshape(*self.env.action_space.shape)

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        call_start = time.perf_counter()
        if self._action_buffer:
            action = self._action_buffer.popleft()
            self._step_count += 1
            self._action_call_times.append(time.perf_counter() - call_start)
            return self._reshape_for_env(action)

        plan_start = time.perf_counter()
        info = self._prepare_info(info_dict)
        pixels = info["pixels"][:, -1].to(self.device)
        goal = info["goal"][:, -1].to(self.device)
        z_current = self._encode(pixels)

        cache_hit = (
            self.cache_goal
            and self._cached_goal_input is not None
            and self._cached_goal_input.shape == goal.shape
            and torch.equal(self._cached_goal_input, goal)
        )
        if cache_hit:
            z_goal = self._cached_z_goal
        else:
            z_goal = self._encode(goal)
            if self.cache_goal:
                self._cached_goal_input = goal.detach().clone()
                self._cached_z_goal = z_goal

        remaining = max(1, min(self.goal_offset - self._step_count, self.idm.max_horizon))
        steps = torch.full(
            (z_current.shape[0],), remaining, device=self.device, dtype=torch.long
        )
        token_trunk = self.idm.generate(
            z_current,
            z_goal,
            steps,
            num_samples=1,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
        )[:, 0]
        actions = self.tokenizer.token_ids_to_actions(token_trunk.cpu().numpy())
        for step in range(self.execute_horizon):
            self._action_buffer.append(actions[:, step])

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._plan_times.append(time.perf_counter() - plan_start)
        action = self._action_buffer.popleft()
        self._step_count += 1
        self._action_call_times.append(time.perf_counter() - call_start)
        return self._reshape_for_env(action)

    @property
    def avg_plan_time_ms(self) -> float:
        return 0.0 if not self._plan_times else 1000.0 * float(np.mean(self._plan_times))

    @property
    def avg_action_call_time_ms(self) -> float:
        return 0.0 if not self._action_call_times else 1000.0 * float(np.mean(self._action_call_times))

    @property
    def num_replans(self) -> int:
        return len(self._plan_times)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop K-step token IDM evaluation")
    parser.add_argument("--dataset", required=True, choices=DATASET_CONFIGS)
    parser.add_argument("--idm", required=True, help="K-step token IDM checkpoint")
    parser.add_argument("--lewm-checkpoint", default=None)
    parser.add_argument("--partition", choices=["train", "val", "test"], default="test")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--goal-offset", type=int, default=None)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--execute-horizon", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name-override", default=None)
    parser.add_argument("--no-goal-cache", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.num_eval < 1 or args.eval_budget < 1 or args.execute_horizon < 1:
        parser.error("num-eval, eval-budget, and execute-horizon must be positive")
    if args.temperature <= 0:
        parser.error("temperature must be positive")

    import stable_worldmodel as swm

    device = torch.device(args.device)
    idm, tokenizer, checkpoint = load_model_checkpoint(args.idm, device)
    manifest = EpisodeSplitManifest.from_dict(checkpoint["split_manifest"])
    goal_offset = (
        checkpoint["dataset_config"]["goal_offset"]
        if args.goal_offset is None
        else args.goal_offset
    )
    if goal_offset < idm.action_horizon:
        parser.error("goal-offset must be at least the trained action horizon")
    if args.execute_horizon > idm.action_horizon:
        parser.error("execute-horizon cannot exceed the trained action horizon")

    dataset_config = DATASET_CONFIGS[args.dataset]
    if args.dataset_name_override:
        dataset_config = dataclasses.replace(
            dataset_config, dataset_name=args.dataset_name_override
        )
    dataset = swm.data.HDF5Dataset(
        dataset_config.dataset_name, keys_to_cache=dataset_config.cache_keys
    )
    transform = {
        "pixels": get_img_transform(dataset_config.img_size),
        "goal": get_img_transform(dataset_config.img_size),
    }
    process = build_normalizers(dataset, dataset_config.cache_keys)
    episodes, starts = sample_eval_starts(
        dataset,
        manifest.episode_ids(args.partition),
        num_eval=args.num_eval,
        goal_offset=goal_offset,
        seed=args.seed,
    )

    data_root = os.environ.get("STABLEWM_HOME", "./stable-wm-data")
    lewm_path = args.lewm_checkpoint or os.path.join(
        data_root, "checkpoints", args.dataset, "lewm"
    )
    lewm = load_lewm_model(lewm_path, str(device))
    jepa = lewm.model if hasattr(lewm, "model") else lewm
    jepa.eval().requires_grad_(False)
    policy = KStepTokenIDMPolicy(
        jepa,
        idm,
        tokenizer,
        goal_offset=goal_offset,
        execute_horizon=args.execute_horizon,
        process=process,
        transform=transform,
        device=device,
        cache_goal=not args.no_goal_cache,
        do_sample=args.sample,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    world = swm.World(
        env_name=dataset_config.env,
        num_envs=len(episodes),
        image_shape=(dataset_config.img_size, dataset_config.img_size),
        max_episode_steps=2 * max(args.eval_budget, goal_offset) + 5,
        **(dataset_config.world_kwargs or {}),
    )
    world.set_policy(policy)

    started = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=starts.tolist(),
        goal_offset=goal_offset,
        eval_budget=args.eval_budget,
        episodes_idx=episodes.tolist(),
        callables=dataset_config.callables,
    )
    elapsed = time.time() - started
    result = {
        "checkpoint": str(Path(args.idm).resolve()),
        "partition": args.partition,
        "num_eval": len(episodes),
        "goal_offset": goal_offset,
        "eval_budget": args.eval_budget,
        "action_horizon": idm.action_horizon,
        "execute_horizon": args.execute_horizon,
        "sampling": args.sample,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "episodes": episodes.tolist(),
        "start_steps": starts.tolist(),
        "metrics": _json_safe(metrics),
        "wall_clock_seconds": elapsed,
        "milliseconds_per_episode": 1000.0 * elapsed / len(episodes),
        "average_replan_ms": policy.avg_plan_time_ms,
        "average_action_call_ms": policy.avg_action_call_time_ms,
        "num_replans": policy.num_replans,
    }
    print(json.dumps(result, indent=2))
    output_path = Path(args.output) if args.output else Path(args.idm).resolve().parent / (
        f"closed_loop_{args.partition}_exec{args.execute_horizon}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
