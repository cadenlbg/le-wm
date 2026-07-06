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
                "checkpoint_every": 50,
                "val_every": 1,
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

    best_val = float("inf")
    global_step = 0
    for epoch in range(int(cfg.training.num_epochs)):
        model.train()
        train_losses = []
        iterator = tqdm(train_loader, desc=f"Training epoch {epoch}", leave=False, mininterval=float(cfg.training.tqdm_interval_sec)) if tqdm else train_loader
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(iterator):
            batch = move_batch(batch, device)
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
        print(record)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
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
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            torch.save(save_payload, ckpt_dir / "best.pt")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    import sys

    run(OmegaConf.from_cli(sys.argv[1:]))
