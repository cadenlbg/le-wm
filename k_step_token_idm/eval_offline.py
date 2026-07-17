#!/usr/bin/env python3
"""Held-out embedding-level evaluation for K-step token IDM."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from k_step_token_idm.dataset import EmbeddingStore, KStepEmbeddingDataset
from k_step_token_idm.metrics import evaluate_offline
from k_step_token_idm.model import AutoregressiveKStepTokenIDM, KStepTokenIDMConfig
from k_step_token_idm.splits import EpisodeSplitManifest, load_episode_split
from single_step_token_idm.tokenization import ActionTokenizer


def load_model_checkpoint(path: str, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = AutoregressiveKStepTokenIDM(KStepTokenIDMConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    tokenizer = ActionTokenizer.from_dict(payload["tokenizer"])
    return model, tokenizer, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline K-step token IDM evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--partition", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer, payload = load_model_checkpoint(args.checkpoint, device)
    dataset_cfg = payload["dataset_config"]
    embeddings = args.embeddings or dataset_cfg["embeddings"]
    store = EmbeddingStore(embeddings)
    if args.split_manifest:
        manifest = load_episode_split(args.split_manifest, store.unique_episode_ids)
    else:
        manifest = EpisodeSplitManifest.from_dict(payload["split_manifest"])
        manifest.validate(store.unique_episode_ids)

    dataset = KStepEmbeddingDataset(
        store,
        manifest,
        args.partition,
        action_horizon=dataset_cfg["action_horizon"],
        goal_offset=dataset_cfg["goal_offset"],
        goal_sampling="fixed" if args.partition != "train" else dataset_cfg["goal_sampling"],
        max_goal_horizon=dataset_cfg["max_goal_horizon"],
        tokenizer=tokenizer,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    metrics = evaluate_offline(model, loader, tokenizer, device)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "embeddings": str(Path(embeddings).resolve()),
        "partition": args.partition,
        "num_samples": len(dataset),
        "num_episodes": len(manifest.episode_ids(args.partition)),
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
