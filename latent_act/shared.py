import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms


def experiments_root() -> Path:
    if os.environ.get("LEWM_EXPERIMENTS_DIR"):
        return Path(os.environ["LEWM_EXPERIMENTS_DIR"])
    if os.environ.get("STABLEWM_HOME"):
        return Path(os.environ["STABLEWM_HOME"]).expanduser().resolve().parent / "experiments"
    return Path("/data/zflin/lewm_re/experiments")


def datasets_root() -> Path:
    if os.environ.get("LEWM_DATASETS_DIR"):
        return Path(os.environ["LEWM_DATASETS_DIR"])
    if os.environ.get("STABLEWM_HOME"):
        return Path(os.environ["STABLEWM_HOME"]).expanduser().resolve() / "latent_bc_datasets"
    return Path("/data/zflin/lewm_re/stablewm_data/latent_bc_datasets")


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
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "latent_bc_datasets":
        path = Path(*parts[1:])
    return datasets_root() / path


def episode_column(dataset) -> str:
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def load_hdf5_dataset(dataset_name, cache_dir=None, keys_to_cache: Optional[Iterable[str]] = None):
    keys_to_cache = list(keys_to_cache or ["action", "proprio", "state"])
    dataset_path = Path(cache_dir or swm.data.utils.get_cache_dir())
    if hasattr(swm.data, "HDF5Dataset"):
        return swm.data.HDF5Dataset(
            dataset_name,
            keys_to_cache=keys_to_cache,
            cache_dir=dataset_path,
        )
    return swm.data.load_dataset(
        dataset_name,
        transform=None,
        cache_dir=cache_dir,
        keys_to_cache=keys_to_cache,
    )


def transform_batch(transform, images):
    return torch.stack([transform(img) for img in images], dim=0)


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)

