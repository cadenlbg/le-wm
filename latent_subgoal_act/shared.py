from __future__ import annotations

import os
from pathlib import Path


def experiments_root() -> Path:
    if os.environ.get("LEWM_EXPERIMENTS_DIR"):
        return Path(os.environ["LEWM_EXPERIMENTS_DIR"]).expanduser().resolve()
    if os.environ.get("STABLEWM_HOME"):
        return Path(os.environ["STABLEWM_HOME"]).expanduser().resolve().parent / "experiments"
    return Path("/data/zflin/lewm_re/experiments")


def datasets_root() -> Path:
    if os.environ.get("LEWM_SUBGOAL_DATASETS_DIR"):
        return Path(os.environ["LEWM_SUBGOAL_DATASETS_DIR"]).expanduser().resolve()
    if os.environ.get("STABLEWM_HOME"):
        return Path(os.environ["STABLEWM_HOME"]).expanduser().resolve() / "latent_subgoal_act_datasets"
    return Path("/data/zflin/lewm_re/stablewm_data/latent_subgoal_act_datasets")


def resolve_experiment_path(path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "experiments":
        path = Path(*parts[1:])
    return experiments_root() / path


def resolve_dataset_path(path) -> Path:
    path = Path(path).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        parts = path.parts
        if parts and parts[0] == "latent_subgoal_act_datasets":
            path = Path(*parts[1:])
        candidates.append(datasets_root() / path)
        candidates.append(experiments_root() / "latent_subgoal_act_datasets" / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    stem = path.stem
    for root in (datasets_root(), experiments_root() / "latent_subgoal_act_datasets"):
        if root.exists():
            hits = sorted(root.glob(f"{stem}*.pt"))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"Could not resolve dataset path from {path}. Checked: {candidates}")
