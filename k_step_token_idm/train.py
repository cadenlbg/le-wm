#!/usr/bin/env python3
"""Train the autoregressive K-step token IDM."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from k_step_token_idm.dataset import EmbeddingStore, KStepEmbeddingDataset
from k_step_token_idm.metrics import compute_training_loss, evaluate_offline
from k_step_token_idm.model import AutoregressiveKStepTokenIDM, KStepTokenIDMConfig
from k_step_token_idm.splits import load_or_create_episode_split, save_episode_split
from single_step_token_idm.tokenization import ActionTokenizer, ActionTokenizerConfig


def build_tokenizer(
    store: EmbeddingStore,
    train_episode_ids: list[int],
    *,
    n_bins: int,
    normalization: str,
    token_offset: int,
) -> ActionTokenizer:
    stats = store.action_stats(train_episode_ids, use_q99=normalization == "bounds_q99")
    cfg = ActionTokenizerConfig(
        action_dim=store.actions.shape[1],
        n_bins=n_bins,
        normalization=normalization,
        token_offset=token_offset,
    )
    return ActionTokenizer.from_stats(stats, cfg)


def build_optimizer(model, condition_lr: float, decoder_lr: float, weight_decay: float):
    condition_prefixes = ("horizon_embed", "condition_backbone", "ada_scale", "ada_shift")
    condition_parameters = []
    decoder_parameters = []
    for name, parameter in model.named_parameters():
        target = condition_parameters if name.startswith(condition_prefixes) else decoder_parameters
        target.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": condition_parameters, "lr": condition_lr, "name": "condition"},
            {"params": decoder_parameters, "lr": decoder_lr, "name": "decoder"},
        ],
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer, epochs: int, warmup_epochs: int, min_lr_ratio: float):
    def factor(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return max(1.0 / warmup_epochs, (epoch + 1) / warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    tokenizer,
    manifest,
    args,
    epoch: int,
    best_val_free_l1: float,
    train_metrics: dict,
    val_metrics: dict,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": asdict(model.cfg),
        "tokenizer": tokenizer.to_dict(),
        "split_manifest": manifest.to_dict(),
        "dataset_config": {
            "embeddings": str(Path(args.embeddings).resolve()),
            "action_horizon": args.action_horizon,
            "goal_offset": args.goal_offset,
            "goal_sampling": args.goal_sampling,
            "max_goal_horizon": args.max_goal_horizon,
        },
        "epoch": int(epoch),
        "best_val_free_l1": float(best_val_free_l1),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "parameter_count": model.num_parameters(),
    }


def load_resume(path, model, optimizer, scheduler, tokenizer, manifest, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["config"] != asdict(model.cfg):
        raise ValueError("resume checkpoint model config does not match current arguments")
    if payload["tokenizer"] != tokenizer.to_dict():
        raise ValueError("resume checkpoint tokenizer does not match training tokenizer")
    if payload["split_manifest"] != manifest.to_dict():
        raise ValueError("resume checkpoint split manifest does not match")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    return int(payload["epoch"]), float(payload["best_val_free_l1"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train autoregressive K-step token IDM")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True, help="Experiment directory")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--val-fraction-of-remaining", type=float, default=0.1)
    parser.add_argument("--action-horizon", type=int, default=3)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--goal-sampling", choices=["fixed", "uniform"], default="fixed")
    parser.add_argument("--max-goal-horizon", type=int, default=50)
    parser.add_argument("--n-bins", type=int, default=256)
    parser.add_argument("--normalization", choices=["bounds", "bounds_q99"], default="bounds_q99")
    parser.add_argument("--token-offset", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--condition-layers", type=int, default=3)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-ffn-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", choices=["gelu", "relu", "silu"], default="gelu")
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--condition-lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--l1-coef", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="k-step-token-idm")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.checkpoint_every < 1:
        parser.error("epochs, batch-size, and checkpoint-every must be positive")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("warmup-epochs must be in [0, epochs)")
    if args.condition_lr <= 0 or args.decoder_lr <= 0:
        parser.error("learning rates must be positive")
    if not 0 <= args.min_lr_ratio <= 1:
        parser.error("min-lr-ratio must be in [0, 1]")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    store = EmbeddingStore(args.embeddings)
    split_path = Path(args.split_manifest) if args.split_manifest else output / "split_manifest.json"
    manifest = load_or_create_episode_split(
        split_path,
        store.unique_episode_ids,
        split_seed=args.split_seed,
        test_fraction=args.test_fraction,
        val_fraction_of_remaining=args.val_fraction_of_remaining,
    )
    if split_path.resolve() != (output / "split_manifest.json").resolve():
        save_episode_split(manifest, output / "split_manifest.json")

    tokenizer = build_tokenizer(
        store,
        manifest.train_episode_ids,
        n_bins=args.n_bins,
        normalization=args.normalization,
        token_offset=args.token_offset,
    )
    common = dict(
        store=store,
        manifest=manifest,
        action_horizon=args.action_horizon,
        goal_offset=args.goal_offset,
        max_goal_horizon=args.max_goal_horizon,
        tokenizer=tokenizer,
        goal_seed=args.seed,
    )
    train_dataset = KStepEmbeddingDataset(
        partition="train", goal_sampling=args.goal_sampling, **common
    )
    val_dataset = KStepEmbeddingDataset(partition="val", goal_sampling="fixed", **common)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    cfg = KStepTokenIDMConfig(
        embed_dim=store.embeddings.shape[1],
        action_dim=store.actions.shape[1],
        n_bins=args.n_bins,
        action_horizon=args.action_horizon,
        hidden_dim=args.hidden_dim,
        condition_layers=args.condition_layers,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ffn_dim=args.transformer_ffn_dim,
        dropout=args.dropout,
        activation=args.activation,
        noise_sigma=args.noise_sigma,
        max_goal_horizon=args.max_goal_horizon,
        token_offset=args.token_offset,
    )
    model = AutoregressiveKStepTokenIDM(cfg).to(device)
    optimizer = build_optimizer(model, args.condition_lr, args.decoder_lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs, args.min_lr_ratio)
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        start_epoch, best_val = load_resume(
            args.resume, model, optimizer, scheduler, tokenizer, manifest, device
        )

    print(f"Model parameters: {model.num_parameters():,}")
    print(
        f"Samples: train={len(train_dataset):,}, val={len(val_dataset):,}; "
        f"episodes={len(manifest.train_episode_ids)}/{len(manifest.val_episode_ids)}/{len(manifest.test_episode_ids)}"
    )
    history = {"train": [], "val": []}
    history_path = output / "history.json"
    if history_path.exists() and args.resume:
        history = json.loads(history_path.read_text(encoding="utf-8"))

    wandb_run = None
    if args.wandb:
        wandb = importlib.import_module("wandb")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={**vars(args), "model": asdict(cfg), "parameter_count": model.num_parameters()},
        )

    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        sums = {"loss": 0.0, "ce": 0.0, "l1": 0.0, "entropy": 0.0}
        batches = 0
        for batch in train_loader:
            z_t = batch["z_t"].to(device)
            z_goal = batch["z_goal"].to(device)
            steps = batch["steps_remaining"].to(device)
            actions = batch["actions"].to(device)
            targets = batch["action_tokens"].to(device)
            logits = model(z_t, z_goal, steps, targets)
            loss, metrics = compute_training_loss(
                logits, targets, actions, tokenizer, args.l1_coef
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            for key in sums:
                sums[key] += metrics[key]
            batches += 1

        train_metrics = {key: value / batches for key, value in sums.items()}
        train_metrics["epoch"] = epoch + 1
        train_metrics["condition_lr"] = optimizer.param_groups[0]["lr"]
        train_metrics["decoder_lr"] = optimizer.param_groups[1]["lr"]
        val_metrics = evaluate_offline(model, val_loader, tokenizer, device)
        val_metrics["epoch"] = epoch + 1
        is_best = val_metrics["free_l1"] < best_val
        if is_best:
            best_val = val_metrics["free_l1"]
        scheduler.step()

        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            tokenizer,
            manifest,
            args,
            epoch + 1,
            best_val,
            train_metrics,
            val_metrics,
        )
        torch.save(payload, output / "last.pt")
        if is_best:
            torch.save(payload, output / "best.pt")
        if (epoch + 1) % args.checkpoint_every == 0:
            torch.save(payload, checkpoint_dir / f"epoch_{epoch + 1:04d}.pt")

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if wandb_run is not None:
            log = {f"train/{k}": v for k, v in train_metrics.items() if k != "epoch"}
            log.update({f"val/{k}": v for k, v in val_metrics.items() if k != "epoch"})
            log["best/val_free_l1"] = best_val
            wandb_run.log(log, step=epoch + 1)
        print(
            f"epoch {epoch + 1:>4d}/{args.epochs} "
            f"train={train_metrics['loss']:.5f} "
            f"val_tf_ce={val_metrics['teacher_ce']:.5f} "
            f"val_free_l1={val_metrics['free_l1']:.5f}"
        )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
