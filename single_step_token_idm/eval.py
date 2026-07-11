#!/usr/bin/env python3
"""Evaluate a trained single-step token IDM on an embedding dataset."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from single_step_token_idm.dataset import TransitionEmbeddingDataset
from single_step_token_idm.model import GoalConditionedTokenIDM, TokenIDMConfig
from single_step_token_idm.tokenization import ActionTokenizer


@torch.no_grad()
def evaluate(model, loader, tokenizer, device: torch.device) -> dict[str, float]:
    model.eval()
    ce_vals, l1_vals, acc_vals = [], [], []
    for batch in loader:
        z_t = batch["z_t"].to(device)
        z_goal = batch["z_goal"].to(device)
        steps = batch["steps_remaining"].to(device)
        raw_actions = batch["action"].to(device)
        targets = batch["action_tokens"].to(device)
        logits = model(z_t, z_goal, steps)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        pred_actions = tokenizer.expected_actions_from_logits(logits)
        l1 = F.l1_loss(pred_actions, raw_actions)
        pred = torch.argmax(logits, dim=-1)
        acc = (pred == targets).float().mean()
        ce_vals.append(float(ce.item()))
        l1_vals.append(float(l1.item()))
        acc_vals.append(float(acc.item()))
    return {"ce": float(np.mean(ce_vals)), "l1": float(np.mean(l1_vals)), "token_acc": float(np.mean(acc_vals))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate single-step token IDM")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = ActionTokenizer.from_dict(ckpt["tokenizer"])
    cfg = TokenIDMConfig(**ckpt["config"])
    model = GoalConditionedTokenIDM(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    dataset = TransitionEmbeddingDataset(args.embeddings, tokenizer=tokenizer)
    dataset.set_tokenizer(tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    metrics = evaluate(model, loader, tokenizer, device)
    print(metrics)


if __name__ == "__main__":
    main()

