#!/usr/bin/env python3
"""Extract frozen LeWM embeddings for K-step token IDM training."""

from __future__ import annotations

import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from single_step_token_idm.dataset import extract_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen LeWM embeddings")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-prefetch", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-q99-stats", action="store_true")
    args = parser.parse_args()
    extract_embeddings(
        checkpoint_path=args.checkpoint,
        h5_path=args.h5,
        output_path=args.output,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_prefetch=args.num_prefetch,
        device=args.device,
        store_q99_stats=not args.no_q99_stats,
    )


if __name__ == "__main__":
    main()
