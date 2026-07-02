#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import stable_worldmodel as swm


COMMON_PREFIXES = ("module.", "model.", "world_model.")


def extract_state_dict(obj):
    if not isinstance(obj, dict):
        return obj
    for key in ("state_dict", "model", "module", "weights"):
        val = obj.get(key)
        if isinstance(val, dict):
            return val
    return obj


def normalize_key(key):
    new_key = key
    for prefix in COMMON_PREFIXES:
        if new_key.startswith(prefix):
            new_key = new_key[len(prefix) :]
    return new_key


def restructure_state_dict(state_dict):
    translated_state_dict = {}

    for key, tensor in state_dict.items():
        new_key = normalize_key(key)

        if "encoder.encoder.layer." in new_key:
            new_key = new_key.replace("encoder.encoder.layer.", "encoder.layers.")
            new_key = new_key.replace("attention.attention.query", "attention.q_proj")
            new_key = new_key.replace("attention.attention.key", "attention.k_proj")
            new_key = new_key.replace("attention.attention.value", "attention.v_proj")
            new_key = new_key.replace("attention.output.dense", "attention.o_proj")
            new_key = new_key.replace("intermediate.dense", "mlp.fc1")
            new_key = new_key.replace("output.dense", "mlp.fc2")

        translated_state_dict[new_key] = tensor

    return translated_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF repo local folder under $STABLEWM_HOME")
    parser.add_argument("--run-name", required=True, help="Output run name, e.g. pusht/lewm")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to the loader cache layout under $STABLEWM_HOME/checkpoints/",
    )
    parser.add_argument(
        "--legacy-object",
        action="store_true",
        help="Also save a legacy *_object.ckpt file next to the loader cache layout.",
    )
    args = parser.parse_args()

    root = Path(swm.data.utils.get_cache_dir())
    src = root / args.repo
    cache_name = args.run_name.replace("/", "--")
    cache_dir = root / "checkpoints" / f"models--{cache_name}"
    out = Path(args.output) if args.output else cache_dir / "weights.pt"

    cfg_path = src / "config.json"
    weights_path = src / "weights.pt"

    cfg = OmegaConf.create(json.loads(cfg_path.read_text()))
    model = instantiate(cfg)

    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = restructure_state_dict(extract_state_dict(payload))

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out)
    print(f"saved to: {out}")

    if args.output is None:
        shutil.copy2(cfg_path, cache_dir / "config.json")
        print(f"saved to: {cache_dir / 'config.json'}")

    if args.legacy_object:
        legacy_out = root / f"{args.run_name}_object.ckpt"
        legacy_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd, legacy_out)
        print(f"saved to: {legacy_out}")


if __name__ == "__main__":
    main()
