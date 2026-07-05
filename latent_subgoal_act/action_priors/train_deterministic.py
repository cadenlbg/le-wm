from __future__ import annotations

from datetime import date
import json
import math

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from latent_subgoal_act.action_priors.common import GoalActionDataset, episode_split, move_batch, save_policy_checkpoint
from latent_subgoal_act.action_priors.deterministic_model import GoalConditionedActionPrior
from latent_subgoal_act.shared import resolve_dataset_path, resolve_experiment_path

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def build_default_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "dataset": "latent_subgoal_act_datasets/pusht_fixed_g25_k25_t25_train.pt",
            "output": f"{date.today().isoformat()}_goal_action_prior",
            "seed": 42,
            "train_split": 0.9,
            "device": "cuda",
            "max_samples": None,
            "action_horizon": None,
            "loader": {"batch_size": 256, "num_workers": 0},
            "model": {"hidden_dim": 512, "depth": 4, "dropout": 0.1, "num_heads": 8},
            "optim": {"lr": 3e-4, "weight_decay": 1e-4},
            "train": {"epochs": 100, "grad_clip": 1.0},
            "loss": {"lambda_smooth": 0.0},
            "log": {"print_every_factor": 1.1, "print_first_n": 5},
        }
    )


def _run_epoch(model, loader, optimizer, device, cfg, train):
    model.train(train)
    totals = {}
    n_batches = 0
    iterator = tqdm(loader, desc="train" if train else "eval", leave=False, unit="batch") if tqdm is not None else loader
    for batch in iterator:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            pred_action = model(batch["z_t"], batch["z_g"])
            action_loss = F.mse_loss(pred_action, batch["action"])
            smooth_loss = pred_action[:, 1:].sub(pred_action[:, :-1]).pow(2).mean() if pred_action.size(1) > 1 else pred_action.new_tensor(0.0)
            loss = action_loss + cfg.loss.lambda_smooth * smooth_loss
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
        totals["action_mse"] = totals.get("action_mse", 0.0) + float(action_loss.detach())
        totals["smooth"] = totals.get("smooth", 0.0) + float(smooth_loss.detach())
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def run(cfg: DictConfig):
    cfg = OmegaConf.merge(build_default_cfg(), cfg)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    payload = torch.load(resolve_dataset_path(cfg.dataset), map_location="cpu", weights_only=False)
    available_action_horizon = int(payload["action"].shape[1])
    action_horizon = available_action_horizon if cfg.action_horizon is None else int(cfg.action_horizon)
    cfg.action_horizon = action_horizon
    dataset = GoalActionDataset(payload, action_horizon=action_horizon, max_samples=cfg.max_samples)
    metadata = payload["metadata"]
    train_idx, val_idx = episode_split(payload["episode"][: len(dataset)], cfg.train_split, cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, generator=generator, **cfg.loader)
    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **cfg.loader)

    model_config = {
        "latent_dim": metadata["latent_dim"],
        "action_dim": metadata["action_dim"],
        "action_horizon": action_horizon,
        **OmegaConf.to_container(cfg.model, resolve=True),
    }
    model = GoalConditionedActionPrior(**model_config).to(device)
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
        if epoch <= int(cfg.log.print_first_n) or epoch >= next_print_epoch:
            print(record)
            next_print_epoch = max(epoch + 1, int(math.ceil(max(epoch, 1) * float(cfg.log.print_every_factor))))
        val_score = val_metrics["action_mse"]
        if val_score < best_val:
            best_val = val_score
            save_policy_checkpoint(output / "policy.pt", model, OmegaConf.to_container(cfg, resolve=True), metadata, epoch, best_val, model_config)


if __name__ == "__main__":
    import sys

    run(OmegaConf.from_cli(sys.argv[1:]))

