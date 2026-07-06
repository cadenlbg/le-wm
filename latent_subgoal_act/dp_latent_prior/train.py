from __future__ import annotations

from datetime import date
import json
import math

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from latent_subgoal_act.action_priors.common import GoalActionDataset, episode_split, move_batch
from latent_subgoal_act.dp_latent_prior.common import ActionNormalizer, EMAModel, save_dp_checkpoint
from latent_subgoal_act.dp_latent_prior.scheduler import DDPMScheduler
from latent_subgoal_act.dp_latent_prior.temporal_unet import LatentTemporalUnet
from latent_subgoal_act.shared import resolve_dataset_path, resolve_experiment_path

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def build_default_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "dataset": "latent_subgoal_act_datasets/pusht_fixed_g25_k25_t25_train.pt",
            "output": f"{date.today().isoformat()}_dp_latent_prior",
            "seed": 42,
            "train_split": 0.9,
            "device": "cuda",
            "max_samples": None,
            "prediction_horizon": None,
            "loader": {"batch_size": 256, "num_workers": 0},
            "model": {
                "down_dims": [256, 512, 1024],
                "diffusion_step_embed_dim": 128,
                "kernel_size": 5,
                "n_groups": 8,
            },
            "diffusion": {"num_steps": 100, "beta_schedule": "cosine", "beta_start": 1e-4, "beta_end": 2e-2},
            "normalizer": {"enabled": False},
            "ema": {"enabled": True, "decay": 0.995},
            "optim": {"lr": 1e-4, "weight_decay": 1e-6},
            "train": {"epochs": 400, "grad_clip": 1.0},
            "log": {"print_every_factor": 1.1, "print_first_n": 5},
        }
    )


def _run_epoch(model, schedule, normalizer, loader, optimizer, ema, device, cfg, train):
    model.train(train)
    totals = {}
    n_batches = 0
    iterator = tqdm(loader, desc="train" if train else "eval", leave=False, unit="batch") if tqdm is not None else loader
    for batch in iterator:
        batch = move_batch(batch, device)
        action = normalizer.normalize(batch["action"])
        batch_size = action.shape[0]
        step = torch.randint(0, int(cfg.diffusion.num_steps), (batch_size,), device=device)
        noise = torch.randn_like(action)
        noisy_action = schedule.add_noise(action, step, noise)
        with torch.set_grad_enabled(train):
            pred_noise = model(noisy_action, step, batch["z_t"], batch["z_g"])
            loss = F.mse_loss(pred_noise, noise)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
            optimizer.step()
            if ema is not None:
                ema.update(model)
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
        totals["noise_mse"] = totals.get("noise_mse", 0.0) + float(loss.detach())
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def run(cfg: DictConfig):
    cfg = OmegaConf.merge(build_default_cfg(), cfg)
    torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    payload = torch.load(resolve_dataset_path(cfg.dataset), map_location="cpu", weights_only=False)
    available_horizon = int(payload["action"].shape[1])
    prediction_horizon = available_horizon if cfg.prediction_horizon is None else int(cfg.prediction_horizon)
    cfg.prediction_horizon = prediction_horizon
    dataset = GoalActionDataset(payload, action_horizon=prediction_horizon, max_samples=cfg.max_samples)
    metadata = payload["metadata"]
    train_idx, val_idx = episode_split(payload["episode"][: len(dataset)], cfg.train_split, cfg.seed)
    generator = torch.Generator().manual_seed(int(cfg.seed))
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, generator=generator, **cfg.loader)
    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **cfg.loader)

    normalizer = ActionNormalizer.from_actions(
        payload["action"][: len(dataset), :prediction_horizon],
        enabled=bool(cfg.normalizer.enabled),
    ).to(device)
    model_config = {
        "latent_dim": metadata["latent_dim"],
        "action_dim": metadata["action_dim"],
        "prediction_horizon": prediction_horizon,
        **OmegaConf.to_container(cfg.model, resolve=True),
    }
    model = LatentTemporalUnet(**model_config).to(device)
    schedule = DDPMScheduler(**OmegaConf.to_container(cfg.diffusion, resolve=True), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optim)
    ema = EMAModel(model, decay=float(cfg.ema.decay)).to(device) if bool(cfg.ema.enabled) else None

    output = resolve_experiment_path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "config.yaml")
    metrics_path = output / "metrics.jsonl"

    best_val = float("inf")
    next_print_epoch = 1
    for epoch in range(1, int(cfg.train.epochs) + 1):
        train_metrics = _run_epoch(model, schedule, normalizer, train_loader, optimizer, ema, device, cfg, True)
        eval_model = ema.model if ema is not None else model
        val_metrics = _run_epoch(eval_model, schedule, normalizer, val_loader, None, None, device, cfg, False)
        record = {
            "epoch": epoch,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if epoch <= int(cfg.log.print_first_n) or epoch >= next_print_epoch:
            print(record)
            next_print_epoch = max(epoch + 1, int(math.ceil(max(epoch, 1) * float(cfg.log.print_every_factor))))
        val_score = val_metrics["noise_mse"]
        if val_score < best_val:
            best_val = val_score
            ckpt_config = {**model_config, "diffusion": OmegaConf.to_container(cfg.diffusion, resolve=True)}
            save_dp_checkpoint(
                output / "policy.pt",
                model,
                ema.model if ema is not None else None,
                OmegaConf.to_container(cfg, resolve=True),
                metadata,
                epoch,
                best_val,
                ckpt_config,
                normalizer,
            )


if __name__ == "__main__":
    import sys

    run(OmegaConf.from_cli(sys.argv[1:]))

