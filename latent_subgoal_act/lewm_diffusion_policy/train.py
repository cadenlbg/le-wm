from __future__ import annotations

from datetime import date
import json
import math

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset

from latent_subgoal_act.action_priors.common import episode_split, move_batch
from latent_subgoal_act.lewm_diffusion_policy.model import DDPMScheduler, EMAModel, LeWMLatentDiffusionPolicy, LinearActionNormalizer
from latent_subgoal_act.shared import resolve_dataset_path, resolve_experiment_path

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


class LeWMDiffusionDataset(Dataset):
    def __init__(self, payload, horizon: int, history_size: int = 2, max_samples=None):
        self.payload = payload
        self.horizon = int(horizon)
        self.history_size = int(history_size)
        available = int(payload["action"].shape[1])
        if self.horizon > available:
            raise ValueError(f"Requested horizon={self.horizon}, but dataset only has {available} action steps.")
        self.length = int(payload["z_t"].shape[0])
        if max_samples is not None:
            self.length = min(self.length, int(max_samples))
        self.index_by_episode_step = {
            (int(ep), int(step)): idx
            for idx, (ep, step) in enumerate(zip(payload["episode"][: self.length].tolist(), payload["step"][: self.length].tolist()))
        }

    def __len__(self):
        return self.length

    def _history_indices(self, idx):
        ep = int(self.payload["episode"][idx])
        step = int(self.payload["step"][idx])
        indices = []
        for offset in range(self.history_size - 1, -1, -1):
            candidate = self.index_by_episode_step.get((ep, step - offset), idx)
            indices.append(candidate)
        return indices

    def __getitem__(self, idx):
        hist_idx = self._history_indices(idx)
        return {
            "z_t": self.payload["z_t"][idx],
            "z_history": self.payload["z_t"][hist_idx],
            "z_g": self.payload["z_g"][idx],
            "action": self.payload["action"][idx, : self.horizon],
            "episode": self.payload["episode"][idx],
        }


def build_default_cfg():
    return OmegaConf.create(
        {
            "dataset": "latent_subgoal_act_datasets/pusht_fixed_g25_k25_t25_train.pt",
            "output": f"{date.today().isoformat()}_lewm_dp",
            "seed": 42,
            "train_split": 0.9,
            "device": "cuda",
            "max_samples": None,
            "horizon": 16,
            "n_action_steps": 8,
            "history_size": 2,
            "goal_condition": True,
            "dataloader": {"batch_size": 64, "num_workers": 0, "pin_memory": True, "shuffle": True},
            "val_dataloader": {"batch_size": 64, "num_workers": 0, "pin_memory": True, "shuffle": False},
            "policy": {
                "num_inference_steps": 100,
                "diffusion_step_embed_dim": 128,
                "down_dims": [512, 1024, 2048],
                "kernel_size": 5,
                "n_groups": 8,
                "cond_predict_scale": True,
            },
            "noise_scheduler": {
                "num_train_timesteps": 100,
                "beta_schedule": "squaredcos_cap_v2",
                "beta_start": 1e-4,
                "beta_end": 2e-2,
                "clip_sample": True,
            },
            "normalizer": {"enabled": True},
            "optimizer": {"lr": 1e-4, "weight_decay": 1e-6, "betas": [0.95, 0.999], "eps": 1e-8},
            "training": {
                "num_epochs": 3050,
                "gradient_accumulate_every": 1,
                "lr_scheduler": "cosine",
                "lr_warmup_steps": 500,
                "use_ema": True,
                "rollout_every": 50,
                "checkpoint_every": 100,
                "val_every": 1,
                "sample_every": 5,
                "max_train_steps": None,
                "max_val_steps": None,
                "tqdm_interval_sec": 1.0,
                "grad_clip": 1.0,
            },
            "ema": {"update_after_step": 0, "inv_gamma": 1.0, "power": 0.75, "min_value": 0.0, "max_value": 0.9999},
            "logging": {
                "mode": "disabled",
                "project": "lewm_diffusion_policy",
                "name": None,
                "tags": ["lewm", "diffusion_policy", "pusht"],
            },
            "rollout": {
                "enabled": False,
                "num_eval": 6,
                "num_vis": 4,
                "goal_offset_steps": 25,
                "eval_budget": 50,
                "img_size": 224,
                "dataset_name": "pusht_expert_train",
                "lewm_policy": "pusht/lewm",
                "env_name": "swm/PushT-v1",
                "execution_horizon": 8,
                "sample_num_candidates": 8,
                "test_start_seed": 100000,
                "fps": 10,
                "cem_enabled": False,
                "cem_diffusion_topk": 8,
                "cem_num_iters": 3,
                "cem_num_candidates": 32,
                "cem_elite_frac": 0.25,
                "cem_min_std": 0.05,
                "cem_std_scale": 1.0,
            },
        }
    )


def _make_lr_scheduler(optimizer, cfg, total_steps):
    warmup = int(cfg.training.lr_warmup_steps)
    if cfg.training.lr_scheduler == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    def lr_lambda(step):
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = float(step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _run_validation(model, scheduler, normalizer, loader, device, cfg):
    model.eval()
    losses = []
    iterator = tqdm(loader, desc="Validation", leave=False, mininterval=float(cfg.training.tqdm_interval_sec)) if tqdm else loader
    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            batch = move_batch(batch, device)
            losses.append(model.compute_loss(batch, scheduler, normalizer).detach())
            if cfg.training.max_val_steps is not None and batch_idx >= int(cfg.training.max_val_steps) - 1:
                break
    if not losses:
        return None
    return torch.stack(losses).mean().item()


def _wandb_key(prefix: str, key: str) -> str:
    return key if key.startswith(prefix + "/") else f"{prefix}/{key}"


def _as_scalar(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.reshape(-1)[0])
    return None


def _as_float_sequence(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        return None
    scalars = []
    for item in value:
        scalar = _as_scalar(item)
        if scalar is None:
            return None
        scalars.append(scalar)
    return scalars


def _iter_video_paths(value):
    suffixes = {".mp4", ".gif", ".webm"}
    if isinstance(value, (str, bytes)):
        path = str(value)
        if any(path.lower().endswith(suffix) for suffix in suffixes):
            yield path
        return
    if hasattr(value, "__fspath__"):
        path = value.__fspath__()
        if any(str(path).lower().endswith(suffix) for suffix in suffixes):
            yield path
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_video_paths(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_video_paths(item)


def _find_score_sequence(metrics):
    if not isinstance(metrics, dict):
        return None
    preferred_keys = (
        "max_rewards",
        "max_reward",
        "scores",
        "score",
        "rewards",
        "reward",
        "successes",
        "success",
    )
    for key in preferred_keys:
        if key in metrics:
            sequence = _as_float_sequence(metrics[key])
            if sequence:
                return sequence
    for key, value in metrics.items():
        if any(name in str(key).lower() for name in ("reward", "score", "success")):
            sequence = _as_float_sequence(value)
            if sequence:
                return sequence
    return None


def _run_rollout(eval_model, scheduler, normalizer, metadata, cfg, output, device, epoch):
    import wandb
    import stable_worldmodel as swm

    from eval import get_dataset, get_episodes_length, img_transform
    from latent_subgoal_act.lewm_diffusion_policy.eval import LeWMDiffusionWorldPolicy

    rollout_dir = output / "rollouts" / f"epoch_{int(epoch):04d}"
    media_dir = rollout_dir / "media"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = OmegaConf.create(
        {
            "cache_dir": None,
            "dataset": {"keys_to_cache": ["action", "proprio", "state"]},
            "eval": {
                "img_size": int(cfg.rollout.img_size),
                "dataset_name": cfg.rollout.dataset_name,
                "goal_offset_steps": int(cfg.rollout.goal_offset_steps),
                "eval_budget": int(cfg.rollout.eval_budget),
                "callables": [
                    {"method": "_set_state", "args": {"state": {"value": "state"}}},
                    {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
                ],
            },
        }
    )
    world = swm.World(
        env_name=cfg.rollout.env_name,
        num_envs=int(cfg.rollout.num_eval),
        max_episode_steps=2 * int(cfg.rollout.eval_budget),
        image_shape=(224, 224),
    )
    transform = {"pixels": img_transform(eval_cfg), "goal": img_transform(eval_cfg)}
    dataset = get_dataset(eval_cfg, cfg.rollout.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    lewm = swm.wm.utils.load_pretrained(cfg.rollout.lewm_policy).to(device).eval().requires_grad_(False)
    lewm.interpolate_pos_encoding = True
    cem_cfg = OmegaConf.create(
        {
            "enabled": bool(cfg.rollout.cem_enabled),
            "diffusion_topk": int(cfg.rollout.cem_diffusion_topk),
            "num_iters": int(cfg.rollout.cem_num_iters),
            "num_candidates": int(cfg.rollout.cem_num_candidates),
            "elite_frac": float(cfg.rollout.cem_elite_frac),
            "min_std": float(cfg.rollout.cem_min_std),
            "std_scale": float(cfg.rollout.cem_std_scale),
        }
    )
    policy = LeWMDiffusionWorldPolicy(
        lewm_encoder=lewm,
        policy=eval_model,
        scheduler=scheduler,
        normalizer=normalizer,
        transform=transform,
        action_mean=metadata.get("action_mean"),
        action_scale=metadata.get("action_scale"),
        device=str(device),
        execution_horizon=int(cfg.rollout.execution_horizon),
        num_candidates=int(cfg.rollout.sample_num_candidates),
        sample_seed=int(cfg.seed) + int(epoch),
        cem_cfg=cem_cfg,
    )

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - int(cfg.rollout.goal_offset_steps) - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    rng = np.random.default_rng(int(cfg.seed) + int(epoch))
    random_episode_indices = np.sort(rng.choice(valid_indices, size=int(cfg.rollout.num_eval), replace=False))
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]
    world.set_policy(policy)
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=int(cfg.rollout.goal_offset_steps),
        eval_budget=int(cfg.rollout.eval_budget),
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(eval_cfg.eval.callables, resolve=True),
        video=media_dir,
    )

    log = {
        "test/rollout_epoch": int(epoch),
        "test/video_dir": str(media_dir),
    }
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            scalar = _as_scalar(value)
            if scalar is not None:
                log[_wandb_key("test", str(key))] = scalar

    score_sequence = _find_score_sequence(metrics)
    if score_sequence:
        for idx, score in enumerate(score_sequence[: int(cfg.rollout.num_eval)]):
            seed = int(cfg.rollout.test_start_seed) + idx
            log[f"test/sim_max_reward_{seed}"] = float(score)
        if "test/mean_score" not in log:
            log["test/mean_score"] = float(np.mean(score_sequence))

    video_exts = ("*.mp4", "*.gif", "*.webm")
    videos = list(_iter_video_paths(metrics))
    for pattern in video_exts:
        videos.extend(str(path) for path in sorted(media_dir.rglob(pattern)))
        videos.extend(str(path) for path in sorted(rollout_dir.rglob(pattern)))
    videos = sorted(set(videos))[: int(cfg.rollout.num_vis)]
    log["test/num_videos"] = len(videos)
    for idx, video_path in enumerate(videos):
        suffix = str(video_path).rsplit(".", 1)[-1].lower() if "." in str(video_path) else "mp4"
        seed = int(cfg.rollout.test_start_seed) + idx
        log[f"test/sim_video_{seed}"] = wandb.Video(str(video_path), fps=int(cfg.rollout.fps), format=suffix)
    return log


def run(cfg: DictConfig):
    cfg = OmegaConf.merge(build_default_cfg(), cfg)
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    payload = torch.load(resolve_dataset_path(cfg.dataset), map_location="cpu", weights_only=False)
    available_horizon = int(payload["action"].shape[1])
    horizon = available_horizon if cfg.horizon is None else int(cfg.horizon)
    cfg.horizon = horizon
    dataset = LeWMDiffusionDataset(payload, horizon=horizon, history_size=int(cfg.history_size), max_samples=cfg.max_samples)
    metadata = payload["metadata"]
    train_idx, val_idx = episode_split(payload["episode"][: len(dataset)], cfg.train_split, int(cfg.seed))
    generator = torch.Generator().manual_seed(int(cfg.seed))
    train_loader = DataLoader(Subset(dataset, train_idx), generator=generator, **cfg.dataloader)
    val_loader = DataLoader(Subset(dataset, val_idx), **cfg.val_dataloader)

    normalizer = LinearActionNormalizer.fit(payload["action"][: len(dataset), :horizon], enabled=bool(cfg.normalizer.enabled)).to(device)
    model_config = {
        "latent_dim": metadata["latent_dim"],
        "action_dim": metadata["action_dim"],
        "horizon": horizon,
        "n_action_steps": int(cfg.n_action_steps),
        "history_size": int(cfg.history_size),
        "goal_condition": bool(cfg.goal_condition),
        **OmegaConf.to_container(cfg.policy, resolve=True),
    }
    model = LeWMLatentDiffusionPolicy(**model_config).to(device)
    ema = EMAModel(model, **OmegaConf.to_container(cfg.ema, resolve=True)).to(device) if bool(cfg.training.use_ema) else None
    scheduler = DDPMScheduler(**OmegaConf.to_container(cfg.noise_scheduler, resolve=True)).to(device)
    optim_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
    optimizer = torch.optim.AdamW(model.parameters(), **optim_cfg)
    total_steps = max(1, (len(train_loader) * int(cfg.training.num_epochs)) // int(cfg.training.gradient_accumulate_every))
    lr_scheduler = _make_lr_scheduler(optimizer, cfg, total_steps)

    output = resolve_experiment_path(cfg.output)
    ckpt_dir = output / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "config.yaml")
    metrics_path = output / "logs.json.txt"
    wandb_run = None
    if str(cfg.logging.mode) != "disabled":
        import wandb

        wandb_run = wandb.init(
            dir=str(output),
            project=str(cfg.logging.project),
            name=cfg.logging.name,
            mode=str(cfg.logging.mode),
            tags=list(cfg.logging.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"output_dir": str(output)})

    best_val = float("inf")
    global_step = 0
    train_sampling_batch = None
    num_epochs = int(cfg.training.num_epochs)
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        print(f"[train] epoch {epoch + 1}/{num_epochs} start")
        iterator = (
            tqdm(
                train_loader,
                desc=f"Training epoch {epoch + 1}/{num_epochs}",
                leave=False,
                mininterval=float(cfg.training.tqdm_interval_sec),
            )
            if tqdm
            else train_loader
        )
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(iterator):
            batch = move_batch(batch, device)
            if train_sampling_batch is None:
                train_sampling_batch = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in batch.items()}
            raw_loss = model.compute_loss(batch, scheduler, normalizer)
            loss = raw_loss / int(cfg.training.gradient_accumulate_every)
            loss.backward()
            if global_step % int(cfg.training.gradient_accumulate_every) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
            if ema is not None:
                ema.step(model)
            train_losses.append(float(raw_loss.detach()))
            global_step += 1
            if cfg.training.max_train_steps is not None and batch_idx >= int(cfg.training.max_train_steps) - 1:
                break

        eval_model = ema.averaged_model if ema is not None else model
        val_loss = None
        if epoch % int(cfg.training.val_every) == 0:
            val_loss = _run_validation(eval_model, scheduler, normalizer, val_loader, device, cfg)

        record = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": lr_scheduler.get_last_lr()[0],
            "train_loss": float(np.mean(train_losses)) if train_losses else None,
            "val_loss": val_loss,
        }
        if train_sampling_batch is not None and epoch % int(cfg.training.sample_every) == 0:
            with torch.no_grad():
                sample_batch = move_batch(train_sampling_batch, device)
                sample = eval_model.conditional_sample(
                    sample_batch["z_history"],
                    sample_batch.get("z_g"),
                    scheduler,
                    num_samples=1,
                ).squeeze(1)
                pred_action = normalizer.unnormalize(sample)
                record["train_action_mse_error"] = torch.nn.functional.mse_loss(pred_action, sample_batch["action"]).item()
        if bool(cfg.rollout.enabled) and wandb_run is not None and epoch % int(cfg.training.rollout_every) == 0:
            print(f"[rollout] epoch {epoch + 1}/{num_epochs} start")
            rollout_log = _run_rollout(eval_model, scheduler, normalizer, metadata, cfg, output, device, epoch)
            record.update(rollout_log)
            print(f"[rollout] epoch {epoch + 1}/{num_epochs} done, videos={record.get('test/num_videos', 0)}")
        print(record)
        json_record = {key: value for key, value in record.items() if not value.__class__.__module__.startswith("wandb")}
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(json_record) + "\n")
        if wandb_run is not None:
            wandb_run.log(record, step=global_step)

        save_payload = {
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "model": model.state_dict(),
            "ema_model": ema.averaged_model.state_dict() if ema is not None else None,
            "model_config": model_config,
            "scheduler_config": scheduler.state_config(),
            "normalizer": normalizer.state_dict(),
            "metadata": metadata,
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
        }
        if epoch % int(cfg.training.checkpoint_every) == 0:
            torch.save(save_payload, ckpt_dir / "latest.pt")
            val_tag = "nan" if val_loss is None else f"{val_loss:.6f}"
            torch.save(save_payload, ckpt_dir / f"epoch={epoch:04d}-val_loss={val_tag}.pt")
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            torch.save(save_payload, ckpt_dir / "best.pt")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    import sys

    run(OmegaConf.from_cli(sys.argv[1:]))
