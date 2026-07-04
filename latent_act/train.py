from __future__ import annotations

from datetime import date
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from latent_act.dataset import LatentACTDataset
from latent_act.model import LatentAwareACTPolicy
from latent_act.shared import resolve_dataset_path, resolve_experiment_path

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _episode_split(episodes, train_split, seed):
    generator = torch.Generator().manual_seed(seed)
    unique_episodes = torch.unique(episodes.cpu())
    perm = unique_episodes[torch.randperm(len(unique_episodes), generator=generator)]
    n_train = max(1, int(len(perm) * train_split))
    train_eps = set(perm[:n_train].tolist())
    train_idx, val_idx = [], []
    for idx, episode in enumerate(episodes.tolist()):
        (train_idx if episode in train_eps else val_idx).append(idx)
    if not val_idx:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    return train_idx, val_idx


def _move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _loss(pred_action, pred_latent, batch, cfg):
    action_loss = F.mse_loss(pred_action, batch["action"])
    target = batch["latent_target"]
    if target.ndim == 3:
        target = target[:, -1]
    latent_loss = F.mse_loss(pred_latent[:, -1], target)
    smooth_loss = pred_action[:, 1:].sub(pred_action[:, :-1]).pow(2).mean() if pred_action.size(1) > 1 else pred_action.new_tensor(0.0)
    total = action_loss + cfg.loss.lambda_latent * latent_loss + cfg.loss.lambda_smooth * smooth_loss
    metrics = {
        "loss": total.detach(),
        "action_mse": action_loss.detach(),
        "latent_mse": latent_loss.detach(),
        "smooth": smooth_loss.detach(),
    }
    return total, metrics


def _run_epoch(model, loader, optimizer, device, cfg, train):
    model.train(train)
    totals = {}
    n_batches = 0
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="train" if train else "eval", leave=False, unit="batch")
    for batch in iterator:
        batch = _move_batch(batch, device)
        with torch.set_grad_enabled(train):
            pred_action, pred_latent = model(batch["z_t"], batch["z_g"], batch["delta_z"])
            loss, metrics = _loss(pred_action, pred_latent, batch, cfg)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def build_default_cfg() -> DictConfig:
    defaults = OmegaConf.create(
        {
            "dataset": "latent_bc_datasets/pusht_g25_k5.pt",
            "output": f"{date.today().isoformat()}_pusht_latent_act",
            "seed": 42,
            "train_split": 0.9,
            "device": "cuda",
            "max_samples": None,
            "loader": {"batch_size": 256, "num_workers": 0},
            "model": {
                "hidden_dim": 512,
                "depth": 4,
                "dropout": 0.1,
                "num_heads": 8,
                "latent_horizon": 1,
            },
            "optim": {"lr": 3e-4, "weight_decay": 1e-4},
            "train": {"epochs": 100, "grad_clip": 1.0},
            "log": {"print_every_factor": 1.1, "print_first_n": 5},
            "loss": {"lambda_latent": 0.1, "lambda_smooth": 0.0},
        }
    )
    return defaults


def run(cfg: DictConfig):
    cfg = OmegaConf.merge(build_default_cfg(), cfg)

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dataset_path = resolve_dataset_path(cfg.dataset)
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    dataset = LatentACTDataset(payload, max_samples=cfg.max_samples)
    metadata = payload["metadata"]

    episodes = payload["episode"][: len(dataset)]
    train_idx, val_idx = _episode_split(episodes, cfg.train_split, cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, generator=generator, **cfg.loader)
    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **cfg.loader)

    model = LatentAwareACTPolicy(
        latent_dim=metadata["latent_dim"],
        action_dim=metadata["action_dim"],
        action_horizon=metadata["action_horizon"],
        **cfg.model,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optim)

    output = resolve_experiment_path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "config.yaml")
    metrics_path = output / "metrics.jsonl"

    best_val = float("inf")
    next_print_epoch = 1
    for epoch in range(1, int(cfg.train.epochs) + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, cfg, True)
        val_metrics = _run_epoch(model, val_loader, optimizer, device, cfg, False)
        record = {
            "epoch": epoch,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        should_print = epoch <= int(cfg.log.print_first_n) or epoch >= next_print_epoch
        if should_print:
            print(record)
            next_print_epoch = max(
                epoch + 1,
                int(math.ceil(max(epoch, 1) * float(cfg.log.print_every_factor))),
            )

        if val_metrics["action_mse"] + cfg.loss.lambda_latent * val_metrics["latent_mse"] < best_val:
            best_val = val_metrics["action_mse"] + cfg.loss.lambda_latent * val_metrics["latent_mse"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": {
                        "latent_dim": metadata["latent_dim"],
                        "action_dim": metadata["action_dim"],
                        "action_horizon": metadata["action_horizon"],
                        **OmegaConf.to_container(cfg.model, resolve=True),
                    },
                    "metadata": metadata,
                    "epoch": epoch,
                    "val_score": best_val,
                },
                output / "policy.pt",
            )


if __name__ == "__main__":
    import sys

    from omegaconf import OmegaConf

    cli = OmegaConf.from_cli(sys.argv[1:])
    run(cli)
