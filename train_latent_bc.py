from datetime import date
import json
import os
from pathlib import Path
import sys

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from latent_bc import LatentGoalBCPolicy


def _experiments_root():
    if os.environ.get("LEWM_EXPERIMENTS_DIR"):
        return Path(os.environ["LEWM_EXPERIMENTS_DIR"])
    if os.environ.get("STABLEWM_HOME"):
        return Path(os.environ["STABLEWM_HOME"]).expanduser().resolve().parent / "experiments"
    return Path("/data/zflin/lewm_re/experiments")


def _datasets_root():
    if os.environ.get("LEWM_DATASETS_DIR"):
        return Path(os.environ["LEWM_DATASETS_DIR"])
    if os.environ.get("STABLEWM_HOME"):
        return Path(os.environ["STABLEWM_HOME"]).expanduser().resolve() / "latent_bc_datasets"
    return Path("/data/zflin/lewm_re/stablewm_data/latent_bc_datasets")


def _resolve_experiment_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "experiments":
        path = Path(*parts[1:])
    return _experiments_root() / path


def _resolve_dataset_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "latent_bc_datasets":
        path = Path(*parts[1:])
    return _datasets_root() / path


def _set_default_hydra_dir(job_name):
    if any(arg.startswith("hydra.run.dir=") for arg in sys.argv[1:]):
        return
    run_dir = _experiments_root() / "hydra" / job_name
    sys.argv.append(f"hydra.run.dir={run_dir}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}")


class LatentBCDataset(Dataset):
    def __init__(self, payload):
        self.payload = payload

    def __len__(self):
        return self.payload["z_t"].shape[0]

    def __getitem__(self, idx):
        return {
            "z_t": self.payload["z_t"][idx],
            "z_g": self.payload["z_g"][idx],
            "delta_z": self.payload["delta_z"][idx],
            "action": self.payload["action"][idx],
            "episode": self.payload["episode"][idx],
        }


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


def _loss(pred, target, lambda_smooth, lambda_mag):
    bc_loss = F.mse_loss(pred, target)
    smooth_loss = pred[:, 1:].sub(pred[:, :-1]).pow(2).mean() if pred.size(1) > 1 else pred.new_tensor(0.0)
    mag_loss = pred.pow(2).mean()
    total = bc_loss + lambda_smooth * smooth_loss + lambda_mag * mag_loss
    return total, {
        "loss": total.detach(),
        "bc_mse": bc_loss.detach(),
        "smooth": smooth_loss.detach(),
        "magnitude": mag_loss.detach(),
    }


def _run_epoch(model, loader, optimizer, device, cfg, train):
    model.train(train)
    totals = {}
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            pred = model(batch["z_t"], batch["z_g"], batch["delta_z"])
            loss, metrics = _loss(
                pred,
                batch["action"],
                cfg.loss.lambda_smooth,
                cfg.loss.lambda_mag,
            )
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


@hydra.main(version_base=None, config_path=None)
def run(cfg: DictConfig):
    defaults = OmegaConf.create(
        {
            "dataset": "latent_bc_datasets/pusht_g25_k5.pt",
            "output": f"{date.today().isoformat()}_pusht_latent_bc",
            "seed": 42,
            "train_split": 0.9,
            "device": "cuda",
            "loader": {"batch_size": 256, "num_workers": 0},
            "model": {
                "hidden_dim": 512,
                "depth": 3,
                "dropout": 0.1,
                "architecture": "mlp",
                "num_heads": 8,
            },
            "optim": {"lr": 0.0003, "weight_decay": 0.0001},
            "train": {"epochs": 100, "grad_clip": 1.0},
            "loss": {"lambda_smooth": 0.0, "lambda_mag": 0.0},
        }
    )
    cfg = OmegaConf.merge(defaults, cfg)

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dataset_path = _resolve_dataset_path(cfg.dataset)
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    dataset = LatentBCDataset(payload)
    metadata = payload["metadata"]

    train_idx, val_idx = _episode_split(payload["episode"], cfg.train_split, cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        shuffle=True,
        generator=generator,
        **cfg.loader,
    )
    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **cfg.loader)

    model = LatentGoalBCPolicy(
        latent_dim=metadata["latent_dim"],
        action_dim=metadata["action_dim"],
        action_horizon=metadata["action_horizon"],
        **cfg.model,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optim)

    output = _resolve_experiment_path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "config.yaml")
    metrics_path = output / "metrics.jsonl"

    best_val = float("inf")
    for epoch in range(1, int(cfg.train.epochs) + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, cfg, True)
        val_metrics = _run_epoch(model, val_loader, optimizer, device, cfg, False)
        record = {
            "epoch": epoch,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(record)

        if val_metrics["bc_mse"] < best_val:
            best_val = val_metrics["bc_mse"]
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
                    "val_bc_mse": best_val,
                },
                output / "policy.pt",
            )


if __name__ == "__main__":
    _set_default_hydra_dir("train_latent_bc")
    run()
