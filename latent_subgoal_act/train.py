from __future__ import annotations

from datetime import date
import json
import math

import torch
import torch.nn.functional as F
import stable_worldmodel as swm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from latent_subgoal_act.dataset import LatentSubgoalACTDataset
from latent_subgoal_act.model import LatentSubgoalACTPolicy
from latent_subgoal_act.shared import resolve_dataset_path, resolve_experiment_path
from latent_subgoal_act.wm_rollout import rollout_latent_with_actions

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def build_default_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "dataset": "latent_subgoal_act_datasets/pusht_g25_k5_h5.pt",
            "output": f"{date.today().isoformat()}_pusht_subgoal_act",
            "seed": 42,
            "train_split": 0.9,
            "device": "cuda",
            "max_samples": None,
            "loader": {"batch_size": 256, "num_workers": 0},
            "model": {
                "hidden_dim": 512,
                "subgoal_depth": 3,
                "action_depth": 4,
                "dropout": 0.1,
                "num_heads": 8,
            },
            "optim": {"lr": 3e-4, "weight_decay": 1e-4},
            "train": {"epochs": 100, "grad_clip": 1.0, "teacher_force_subgoal": False},
            "loss": {"lambda_subgoal": 1.0, "lambda_smooth": 0.0},
            "wm": {
                "enabled": True,
                "policy": "pusht/lewm",
                "history_size": 1,
                "lambda_rollout": 0.0,
                "lambda_align": 0.0,
            },
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


def _load_frozen_wm(cfg, device):
    if not cfg.wm.enabled:
        return None
    wm = swm.wm.utils.load_pretrained(cfg.wm.policy)
    wm = wm.to(device).eval()
    wm.requires_grad_(False)
    if hasattr(wm, "interpolate_pos_encoding"):
        wm.interpolate_pos_encoding = True
    return wm


def _loss(pred_action, pred_z_h, batch, cfg, wm_model=None):
    action_loss = F.mse_loss(pred_action, batch["action"])
    subgoal_loss = F.mse_loss(pred_z_h, batch["z_h"])
    smooth_loss = pred_action[:, 1:].sub(pred_action[:, :-1]).pow(2).mean() if pred_action.size(1) > 1 else pred_action.new_tensor(0.0)
    rollout_loss = pred_action.new_tensor(0.0)
    align_loss = pred_action.new_tensor(0.0)
    if wm_model is not None and (cfg.wm.lambda_rollout > 0 or cfg.wm.lambda_align > 0):
        pred_rollout = rollout_latent_with_actions(
            wm_model,
            batch["z_t"],
            pred_action,
            history_size=cfg.wm.history_size,
        )
        rollout_loss = F.mse_loss(pred_rollout, batch["z_h"])
        align_loss = F.mse_loss(pred_rollout, pred_z_h)

    total = (
        action_loss
        + cfg.loss.lambda_subgoal * subgoal_loss
        + cfg.loss.lambda_smooth * smooth_loss
        + cfg.wm.lambda_rollout * rollout_loss
        + cfg.wm.lambda_align * align_loss
    )
    return total, {
        "loss": total.detach(),
        "action_mse": action_loss.detach(),
        "subgoal_mse": subgoal_loss.detach(),
        "smooth": smooth_loss.detach(),
        "wm_rollout_mse": rollout_loss.detach(),
        "wm_align_mse": align_loss.detach(),
    }


def _run_epoch(model, loader, optimizer, device, cfg, train, wm_model=None):
    model.train(train)
    totals = {}
    n_batches = 0
    iterator = tqdm(loader, desc="train" if train else "eval", leave=False, unit="batch") if tqdm is not None else loader
    for batch in iterator:
        batch = _move_batch(batch, device)
        with torch.set_grad_enabled(train):
            pred_action, pred_z_h = model(
                batch["z_t"],
                batch["z_g"],
                z_h_teacher=batch["z_h"],
                teacher_force_subgoal=bool(cfg.train.teacher_force_subgoal),
            )
            loss, metrics = _loss(pred_action, pred_z_h, batch, cfg, wm_model=wm_model)
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
    dataset = LatentSubgoalACTDataset(payload, max_samples=cfg.max_samples)
    metadata = payload["metadata"]
    train_idx, val_idx = _episode_split(payload["episode"][: len(dataset)], cfg.train_split, cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, generator=generator, **cfg.loader)
    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **cfg.loader)

    model = LatentSubgoalACTPolicy(
        latent_dim=metadata["latent_dim"],
        action_dim=metadata["action_dim"],
        action_horizon=metadata["action_horizon"],
        **cfg.model,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optim)
    wm_model = _load_frozen_wm(cfg, device)

    output = resolve_experiment_path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "config.yaml")
    metrics_path = output / "metrics.jsonl"

    best_val = float("inf")
    next_print_epoch = 1
    for epoch in range(1, int(cfg.train.epochs) + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, cfg, True, wm_model=wm_model)
        val_metrics = _run_epoch(model, val_loader, optimizer, device, cfg, False, wm_model=wm_model)
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

        val_score = val_metrics["action_mse"] + cfg.loss.lambda_subgoal * val_metrics["subgoal_mse"]
        if val_score < best_val:
            best_val = val_score
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

    run(OmegaConf.from_cli(sys.argv[1:]))
