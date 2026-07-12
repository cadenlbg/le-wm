#!/usr/bin/env python3
"""Train the single-step token IDM."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from single_step_token_idm.dataset import TransitionEmbeddingDataset
from single_step_token_idm.model import GoalConditionedTokenIDM, TokenIDMConfig
from single_step_token_idm.tokenization import ActionTokenizer, ActionTokenizerConfig


def build_tokenizer(dataset: TransitionEmbeddingDataset, n_bins: int, normalization: str, token_offset: int) -> ActionTokenizer:
    cfg = ActionTokenizerConfig(
        action_dim=dataset.actions.shape[1],
        n_bins=n_bins,
        normalization=normalization,
        token_offset=token_offset,
    )
    return ActionTokenizer.from_stats(dataset.action_stats, cfg)


def compute_loss(
    logits: torch.Tensor,
    token_targets: torch.Tensor,
    raw_actions: torch.Tensor,
    tokenizer: ActionTokenizer,
    l1_coef: float,
    entropy_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), token_targets.reshape(-1))
    pred_actions = tokenizer.expected_actions_from_logits(logits)
    l1 = F.l1_loss(pred_actions, raw_actions)
    ent = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    total = ce + l1_coef * l1 - entropy_coef * ent
    return total, {"ce": float(ce.item()), "l1": float(l1.item()), "entropy": float(ent.item())}


@torch.no_grad()
def evaluate(model, loader, tokenizer, l1_coef: float, entropy_coef: float, device: torch.device) -> dict[str, float]:
    model.eval()
    ce_vals, l1_vals, ent_vals, acc_vals = [], [], [], []
    for batch in loader:
        z_t = batch["z_t"].to(device)
        z_goal = batch["z_goal"].to(device)
        steps = batch["steps_remaining"].to(device)
        raw_actions = batch["action"].to(device)
        targets = batch["action_tokens"].to(device)
        logits = model(z_t, z_goal, steps)
        loss, parts = compute_loss(logits, targets, raw_actions, tokenizer, l1_coef, entropy_coef)
        _ = loss
        pred = torch.argmax(logits, dim=-1)
        acc_vals.append((pred == targets).float().mean().item())
        ce_vals.append(parts["ce"])
        l1_vals.append(parts["l1"])
        ent_vals.append(parts["entropy"])
    return {
        "ce": float(np.mean(ce_vals)),
        "l1": float(np.mean(l1_vals)),
        "entropy": float(np.mean(ent_vals)),
        "token_acc": float(np.mean(acc_vals)),
    }


def build_scheduler(optimizer, scheduler_type: str, epochs: int, warmup_epochs: int, min_lr: float):
    """Build an epoch-level warmup followed by constant or cosine scheduling."""
    if warmup_epochs == 0:
        if scheduler_type == "constant":
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=min_lr
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / warmup_epochs,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    if scheduler_type == "constant":
        main = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=min_lr,
        )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, main],
        milestones=[warmup_epochs],
    )


def checkpoint_path(output: str, suffix: str) -> str:
    path = Path(output)
    return str(path.with_name(f"{path.stem}_{suffix}{path.suffix or '.pt'}"))


def build_checkpoint(
    model,
    optimizer,
    scheduler,
    cfg,
    tokenizer,
    args,
    dataset,
    epoch: int,
    best_val: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": asdict(cfg),
        "tokenizer": tokenizer.to_dict(),
        "epoch": epoch,
        "best_val_ce": best_val,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_split": args.train_split,
        "split_seed": args.split_seed,
        "split_partition": args.split_partition,
        "held_out_episodes": dataset.held_out_episodes,
        "init_checkpoint": args.init_checkpoint,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "lr": args.lr,
        "min_lr": args.min_lr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train single-step token IDM")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-goal-horizon", type=int, default=50)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--train-split", type=float, default=1.0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-partition", default="train", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--scheduler", choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", default="gelu", choices=["gelu", "relu", "silu"])
    parser.add_argument("--n-bins", type=int, default=256)
    parser.add_argument("--normalization", default="bounds_q99", choices=["bounds", "bounds_q99"])
    parser.add_argument("--token-offset", type=int, default=0)
    parser.add_argument("--l1-coef", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--init-checkpoint", default=None, help="Warm-start model weights from a token IDM checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="single-step-token-idm")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-log-artifact", action="store_true", help="Upload the best checkpoint as a wandb artifact")
    args = parser.parse_args()

    if args.epochs < 1 or args.checkpoint_every < 1:
        parser.error("--epochs and --checkpoint-every must be positive")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        parser.error("--warmup-epochs must be in [0, epochs)")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.min_lr is None:
        args.min_lr = args.lr / 100
    if args.min_lr < 0 or args.min_lr > args.lr:
        parser.error("--min-lr must be in [0, lr]")

    device = torch.device(args.device)
    dataset = TransitionEmbeddingDataset(
        args.embeddings,
        max_goal_horizon=args.max_goal_horizon,
        frameskip=args.frameskip,
        train_split=args.train_split,
        split_seed=args.split_seed,
        split_partition=args.split_partition,
    )
    tokenizer = build_tokenizer(dataset, args.n_bins, args.normalization, args.token_offset)
    dataset.set_tokenizer(tokenizer)

    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.split_seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    cfg = TokenIDMConfig(
        embed_dim=dataset.embeddings.shape[1],
        action_dim=dataset.actions.shape[1],
        n_bins=args.n_bins,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        activation=args.activation,
        noise_sigma=args.noise_sigma,
        max_horizon=args.max_goal_horizon,
    )
    model = GoalConditionedTokenIDM(cfg).to(device)
    if args.init_checkpoint is not None:
        init_data = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        saved_cfg = TokenIDMConfig(**init_data["config"])
        saved_tokenizer = ActionTokenizer.from_dict(init_data["tokenizer"])
        if asdict(saved_cfg) != asdict(cfg):
            raise ValueError(
                "--init-checkpoint model config does not match this run: "
                f"saved={asdict(saved_cfg)}, current={asdict(cfg)}"
            )
        if saved_tokenizer.to_dict() != tokenizer.to_dict():
            raise ValueError("--init-checkpoint tokenizer does not match this run")
        model.load_state_dict(init_data["model_state_dict"], strict=True)
        print(f"Warm-started model weights from {args.init_checkpoint}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
    )

    history = {"train": [], "val": [], "config": asdict(cfg), "tokenizer": tokenizer.to_dict()}
    best_val = float("inf")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    wandb_run = None
    if not args.no_wandb:
        if args.wandb_mode is not None:
            os.environ["WANDB_MODE"] = args.wandb_mode
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as exc:
            raise ImportError("wandb is enabled by default. Install wandb or pass --no-wandb.") from exc

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=args.wandb_tags,
            config={
                **vars(args),
                "model": asdict(cfg),
                "tokenizer": tokenizer.to_dict()["cfg"],
                "dataset_size": len(dataset),
                "n_train": n_train,
                "n_val": n_val,
                "held_out_episodes": dataset.held_out_episodes,
            },
        )

    for epoch in range(args.epochs):
        model.train()
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        train_losses = []
        train_ce_vals, train_l1_vals, train_ent_vals = [], [], []
        for batch in train_loader:
            z_t = batch["z_t"].to(device)
            z_goal = batch["z_goal"].to(device)
            steps = batch["steps_remaining"].to(device)
            raw_actions = batch["action"].to(device)
            targets = batch["action_tokens"].to(device)

            logits = model(z_t, z_goal, steps)
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            pred_actions = tokenizer.expected_actions_from_logits(logits)
            l1 = F.l1_loss(pred_actions, raw_actions)
            probs = torch.softmax(logits, dim=-1)
            ent = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            loss = ce + args.l1_coef * l1 - args.entropy_coef * ent

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
            train_ce_vals.append(float(ce.item()))
            train_l1_vals.append(float(l1.item()))
            train_ent_vals.append(float(ent.item()))

        val_metrics = evaluate(model, val_loader, tokenizer, args.l1_coef, args.entropy_coef, device)
        train_loss = float(np.mean(train_losses))
        train_metrics = {
            "epoch": epoch,
            "loss": train_loss,
            "ce": float(np.mean(train_ce_vals)),
            "l1": float(np.mean(train_l1_vals)),
            "entropy": float(np.mean(train_ent_vals)),
            "lr": epoch_lr,
        }
        history["train"].append(train_metrics)
        history["val"].append({"epoch": epoch, **val_metrics})
        is_best = False
        if val_metrics["ce"] < best_val:
            best_val = val_metrics["ce"]
            is_best = True
        scheduler.step()
        checkpoint = build_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            tokenizer=tokenizer,
            args=args,
            dataset=dataset,
            epoch=epoch + 1,
            best_val=best_val,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
        if is_best:
            torch.save(checkpoint, args.output)
        if (epoch + 1) % args.checkpoint_every == 0:
            torch.save(
                checkpoint,
                checkpoint_path(args.output, f"epoch_{epoch + 1:04d}"),
            )
        torch.save(checkpoint, checkpoint_path(args.output, "last"))

        history_path = args.output.replace(".pt", "_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": train_metrics["loss"],
                    "train/ce": train_metrics["ce"],
                    "train/l1": train_metrics["l1"],
                    "train/entropy": train_metrics["entropy"],
                    "val/ce": val_metrics["ce"],
                    "val/l1": val_metrics["l1"],
                    "val/entropy": val_metrics["entropy"],
                    "val/token_acc": val_metrics["token_acc"],
                    "best/val_ce": best_val,
                    "checkpoint/is_best": int(is_best),
                    "lr": train_metrics["lr"],
                },
                step=epoch + 1,
            )
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"epoch {epoch + 1:>4d}/{args.epochs} "
                f"train={train_loss:.6f} val_ce={val_metrics['ce']:.6f} "
                f"val_l1={val_metrics['l1']:.6f} acc={val_metrics['token_acc']:.4f}"
            )

    if wandb_run is not None:
        if args.wandb_log_artifact and os.path.exists(args.output):
            artifact = wandb_run.Artifact("token-idm-checkpoint", type="model")
            artifact.add_file(args.output)
            history_path = args.output.replace(".pt", "_history.json")
            if os.path.exists(history_path):
                artifact.add_file(history_path)
            wandb_run.log_artifact(artifact)
        wandb_run.finish()


if __name__ == "__main__":
    main()
