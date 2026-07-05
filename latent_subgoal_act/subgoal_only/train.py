from __future__ import annotations

from datetime import date
import json
import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from latent_subgoal_act.shared import resolve_dataset_path, resolve_experiment_path
from latent_subgoal_act.subgoal_only.model import GoalConditionedSubgoalPredictor

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


class SubgoalOnlyDataset(Dataset):
    def __init__(self, payload: Dict[str, Any], subgoal_horizon: Optional[int] = None, max_samples: Optional[int] = None):
        required = ("z_t", "z_g", "z_h_seq", "episode")
        for key in required:
            if key not in payload:
                raise KeyError(f"missing required key in payload: {key}")
        available = int(payload["z_h_seq"].shape[1])
        self.subgoal_horizon = available if subgoal_horizon is None else int(subgoal_horizon)
        if self.subgoal_horizon < 1:
            raise ValueError("subgoal_horizon must be >= 1")
        if self.subgoal_horizon > available:
            raise ValueError(f"Requested subgoal_horizon={self.subgoal_horizon}, but dataset only has {available}.")
        self.payload = payload
        self.length = int(payload["z_t"].shape[0])
        if max_samples is not None:
            self.length = min(self.length, int(max_samples))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "z_t": self.payload["z_t"][idx],
            "z_g": self.payload["z_g"][idx],
            "z_h_seq": self.payload["z_h_seq"][idx, : self.subgoal_horizon],
            "episode": self.payload["episode"][idx],
        }


def build_default_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "dataset": "latent_subgoal_act_datasets/pusht_fixed_g25_k25_t25_train.pt",
            "output": f"{date.today().isoformat()}_subgoal_only",
            "seed": 42,
            "train_split": 0.9,
            "device": "cuda",
            "max_samples": None,
            "subgoal_horizon": None,
            "loader": {"batch_size": 256, "num_workers": 0},
            "model": {"hidden_dim": 512, "depth": 4, "dropout": 0.1, "num_heads": 8},
            "optim": {"lr": 3e-4, "weight_decay": 1e-4},
            "train": {"epochs": 100, "grad_clip": 1.0},
            "loss": {"lambda_terminal": 0.0, "lambda_smooth": 0.0},
            "log": {"print_every_factor": 1.1, "print_first_n": 5},
        }
    )


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
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _loss(pred, target, cfg):
    seq_mse = F.mse_loss(pred, target)
    terminal_mse = F.mse_loss(pred[:, -1], target[:, -1])
    smooth = pred[:, 1:].sub(pred[:, :-1]).pow(2).mean() if pred.size(1) > 1 else pred.new_tensor(0.0)
    total = seq_mse + cfg.loss.lambda_terminal * terminal_mse + cfg.loss.lambda_smooth * smooth
    return total, {
        "loss": total.detach(),
        "seq_mse": seq_mse.detach(),
        "terminal_mse": terminal_mse.detach(),
        "smooth": smooth.detach(),
    }


def _run_epoch(model, loader, optimizer, device, cfg, train):
    model.train(train)
    totals = {}
    n_batches = 0
    iterator = tqdm(loader, desc="train" if train else "eval", leave=False, unit="batch") if tqdm is not None else loader
    for batch in iterator:
        batch = _move_batch(batch, device)
        with torch.set_grad_enabled(train):
            pred = model(batch["z_t"], batch["z_g"])
            loss, metrics = _loss(pred, batch["z_h_seq"], cfg)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def run(cfg: DictConfig):
    cfg = OmegaConf.merge(build_default_cfg(), cfg)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    payload = torch.load(resolve_dataset_path(cfg.dataset), map_location="cpu", weights_only=False)
    available = int(payload["z_h_seq"].shape[1])
    subgoal_horizon = available if cfg.subgoal_horizon is None else int(cfg.subgoal_horizon)
    cfg.subgoal_horizon = subgoal_horizon
    dataset = SubgoalOnlyDataset(payload, subgoal_horizon=subgoal_horizon, max_samples=cfg.max_samples)
    metadata = payload["metadata"]
    train_idx, val_idx = _episode_split(payload["episode"][: len(dataset)], cfg.train_split, cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, generator=generator, **cfg.loader)
    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **cfg.loader)

    model_config = {
        "latent_dim": metadata["latent_dim"],
        "subgoal_horizon": subgoal_horizon,
        **OmegaConf.to_container(cfg.model, resolve=True),
    }
    model = GoalConditionedSubgoalPredictor(**model_config).to(device)
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
        val_score = val_metrics["seq_mse"]
        if val_score < best_val:
            best_val = val_score
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": model_config,
                    "metadata": metadata,
                    "epoch": epoch,
                    "val_score": best_val,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                output / "subgoal_predictor.pt",
            )


if __name__ == "__main__":
    import sys

    run(OmegaConf.from_cli(sys.argv[1:]))

