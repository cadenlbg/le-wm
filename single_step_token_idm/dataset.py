"""Dataset and LeWM embedding extraction for single-step token IDM.

The embedding extraction path is adapted from the cloned GC-IDM repo:
`other exp/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models/idm/dataset.py`.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenization import ActionStats, ActionTokenizer, ActionTokenizerConfig, compute_action_stats


def load_lewm_model(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load a LeWM checkpoint in the same formats used by GC-IDM."""
    from hydra.utils import instantiate

    path = Path(checkpoint_path)

    if path.is_file() and path.suffix == ".ckpt":
        model = torch.load(str(path), map_location=device, weights_only=False)
        model.eval()
        return model

    if path.is_dir():
        cfg_file = path / "config.json"
        wts_file = path / "weights.pt"
        if cfg_file.exists() and wts_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            model = instantiate(cfg)
            state_dict = torch.load(str(wts_file), map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            return model

        ckpts = sorted(path.glob("*_object.ckpt"))
        if ckpts:
            model = torch.load(str(ckpts[-1]), map_location=device, weights_only=False)
            model.eval()
            return model

    if "/" in checkpoint_path and not path.exists():
        from huggingface_hub import hf_hub_download

        cfg_file = hf_hub_download(checkpoint_path, "config.json")
        wts_file = hf_hub_download(checkpoint_path, "weights.pt")
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model = instantiate(cfg)
        state_dict = torch.load(wts_file, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        return model

    raise FileNotFoundError(f"Cannot load LeWM checkpoint from {checkpoint_path!r}")


def _get_encoder_projector(model: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
    """GC-IDM-style access to frozen encoder + projector."""
    jepa = model.model if hasattr(model, "model") else model
    encoder = jepa.encoder
    projector = jepa.projector
    return encoder, projector


def _build_image_transform(img_size: int = 224):
    from torchvision.transforms import v2 as transforms
    import stable_pretraining as spt

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


@torch.no_grad()
def extract_embeddings(
    checkpoint_path: str,
    h5_path: str,
    output_path: str,
    img_size: int = 224,
    batch_size: int = 2048,
    num_prefetch: int = 12,
    device: str = "cuda:0",
    store_q99_stats: bool = True,
) -> str:
    """Extract frozen LeWM embeddings and save them alongside actions / episode ids."""
    model = load_lewm_model(checkpoint_path, device)
    encoder, projector = _get_encoder_projector(model)

    class _EncoderProjector(torch.nn.Module):
        def __init__(self, enc, proj):
            super().__init__()
            self.enc = enc
            self.proj = proj

        def forward(self, x):
            out = self.enc(x, interpolate_pos_encoding=True)
            return self.proj(out.last_hidden_state[:, 0])

    enc_proj = _EncoderProjector(encoder, projector).to(device)
    enc_proj.eval()

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and device != "cpu":
        enc_proj = torch.nn.DataParallel(enc_proj)

    print(f"Loading HDF5: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        actions = f["actions"][:] if "actions" in f else f["action"][:]
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]

        if "ep_idx" in f:
            episode_ids = f["ep_idx"][:].astype(np.int64)
        elif "episode_idx" in f:
            episode_ids = f["episode_idx"][:].astype(np.int64)
        else:
            n_frames = f["pixels"].shape[0]
            episode_ids = np.zeros(n_frames, dtype=np.int64)
            for ei in range(len(ep_len)):
                start = ep_offset[ei]
                end = start + ep_len[ei]
                episode_ids[start:end] = ei

        state = f["state"][:] if "state" in f else None
        proprio = f["proprio"][:] if "proprio" in f else None

        pix_dset = f["pixels"]
        n_frames = pix_dset.shape[0]
        is_hwc = pix_dset.ndim == 4 and pix_dset.shape[-1] == 3
        is_chw = pix_dset.ndim == 4 and pix_dset.shape[1] == 3
        if not (is_hwc or is_chw):
            raise ValueError(f"Unexpected pixel shape: {pix_dset.shape}")

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        all_embeddings: list[np.ndarray] = []
        n_batches = (n_frames + batch_size - 1) // batch_size

        def _read_batch(batch_idx: int):
            s = batch_idx * batch_size
            e = min(s + batch_size, n_frames)
            pix = pix_dset[s:e]
            if is_hwc:
                pix = np.transpose(pix, (0, 3, 1, 2))
            return pix

        with ThreadPoolExecutor(max_workers=num_prefetch) as pool:
            futures = {}
            for i in range(min(num_prefetch, n_batches)):
                futures[i] = pool.submit(_read_batch, i)

            t0 = time.time()
            for b in range(n_batches):
                if (b + 1) % 20 == 0 or b == n_batches - 1:
                    elapsed = time.time() - t0
                    fps = (b + 1) * batch_size / max(elapsed, 1e-6)
                    eta = (n_batches - b - 1) * elapsed / max(b + 1, 1)
                    print(f"  batch {b + 1}/{n_batches}  ({fps:.0f} frames/s, ETA {eta:.0f}s)")

                batch_pix = futures[b].result()
                next_b = b + num_prefetch
                if next_b < n_batches:
                    futures[next_b] = pool.submit(_read_batch, next_b)
                del futures[b]

                batch_pix = torch.from_numpy(batch_pix).float().to(device) / 255.0
                batch_pix = (batch_pix - mean) / std
                emb = enc_proj(batch_pix)
                all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    stats = compute_action_stats(actions, use_q99=store_q99_stats)
    save_dict = dict(
        embeddings=embeddings.astype(np.float32),
        actions=actions.astype(np.float32),
        episode_ids=episode_ids.astype(np.int64),
        embed_dim=embeddings.shape[1],
        action_stats=dict(
            action_min=stats.action_min.astype(np.float32),
            action_max=stats.action_max.astype(np.float32),
            action_q01=None if stats.action_q01 is None else stats.action_q01.astype(np.float32),
            action_q99=None if stats.action_q99 is None else stats.action_q99.astype(np.float32),
            action_mask=stats.action_mask.astype(np.bool_),
        ),
    )
    if state is not None:
        save_dict["state"] = state.astype(np.float32)
    if proprio is not None:
        save_dict["proprio"] = proprio.astype(np.float32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    return output_path


class TransitionEmbeddingDataset(Dataset):
    """Dataset of (z_t, z_goal, steps_remaining, action) pairs."""

    def __init__(
        self,
        embeddings_path: str,
        max_goal_horizon: int = 50,
        frameskip: int = 1,
        train_split: float = 1.0,
        split_seed: int = 42,
        split_partition: str = "train",
        tokenizer: Optional[ActionTokenizer] = None,
    ) -> None:
        data = np.load(embeddings_path, allow_pickle=True)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.actions = data["actions"].astype(np.float32)
        self.ep_ids = data["episode_ids"].astype(np.int64)
        self.max_goal_horizon = int(max_goal_horizon)
        self.frameskip = int(frameskip)
        self.tokenizer = tokenizer or self._build_tokenizer_from_npz(data)
        self.action_stats = self._load_stats_from_npz(data)

        if train_split < 1.0:
            unique_eps = np.unique(self.ep_ids)
            n_holdout = max(1, round(len(unique_eps) * (1.0 - train_split)))
            rng = np.random.default_rng(split_seed)
            holdout_eps = set(rng.choice(unique_eps, size=n_holdout, replace=False).tolist())
            if split_partition == "train":
                keep_eps = set(unique_eps.tolist()) - holdout_eps
            else:
                keep_eps = holdout_eps
            ep_mask = np.isin(self.ep_ids, list(keep_eps))
            self.held_out_episodes = sorted(holdout_eps)
        else:
            ep_mask = np.ones(len(self.ep_ids), dtype=bool)
            self.held_out_episodes = []

        n = len(self.ep_ids)
        same_ep = self.ep_ids[:-self.frameskip] == self.ep_ids[self.frameskip :]
        if self.frameskip == 1:
            no_nan = ~np.isnan(self.actions[:-self.frameskip]).any(axis=1)
        else:
            nan_mask = np.isnan(self.actions).any(axis=1)
            kernel = np.ones(self.frameskip, dtype=bool)
            has_nan = np.convolve(nan_mask, kernel, mode="valid")[: n - self.frameskip] > 0
            no_nan = ~has_nan
        valid = same_ep & no_nan & ep_mask[:-self.frameskip]
        self.valid_indices = np.nonzero(valid)[0].tolist()
        self._build_episode_map()

    def _load_stats_from_npz(self, data) -> ActionStats:
        if "action_stats" not in data.files:
            return compute_action_stats(self.actions, use_q99=True)
        stats = data["action_stats"].item()
        return ActionStats(
            action_min=np.asarray(stats["action_min"], dtype=np.float32),
            action_max=np.asarray(stats["action_max"], dtype=np.float32),
            action_q01=None if stats.get("action_q01") is None else np.asarray(stats["action_q01"], dtype=np.float32),
            action_q99=None if stats.get("action_q99") is None else np.asarray(stats["action_q99"], dtype=np.float32),
            action_mask=np.asarray(stats["action_mask"], dtype=bool) if stats.get("action_mask") is not None else None,
        )

    def _build_tokenizer_from_npz(self, data) -> ActionTokenizer:
        stats = self._load_stats_from_npz(data)
        cfg = ActionTokenizerConfig(action_dim=self.actions.shape[1], n_bins=256, normalization="bounds_q99")
        return ActionTokenizer.from_stats(stats, cfg)

    def set_tokenizer(self, tokenizer: ActionTokenizer) -> None:
        self.tokenizer = tokenizer

    def _build_episode_map(self) -> None:
        self.ep_ranges: dict[int, tuple[int, int]] = {}
        current_ep = self.ep_ids[0]
        start = 0
        for i in range(1, len(self.ep_ids)):
            if self.ep_ids[i] != current_ep:
                self.ep_ranges[current_ep] = (start, i)
                current_ep = self.ep_ids[i]
                start = i
        self.ep_ranges[current_ep] = (start, len(self.ep_ids))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = self.valid_indices[idx]
        k = self.frameskip
        z_t = torch.from_numpy(self.embeddings[i]).float()
        action = torch.from_numpy(self.actions[i : i + k].reshape(-1)).float()

        ep = self.ep_ids[i]
        ep_start, ep_end = self.ep_ranges[ep]
        max_future = min(i + self.max_goal_horizon, ep_end - 1)
        if max_future <= i + 1:
            goal_idx = i + 1
        else:
            goal_idx = np.random.randint(i + 1, max_future + 1)

        z_goal = torch.from_numpy(self.embeddings[goal_idx]).float()
        steps_remaining = torch.tensor(goal_idx - i, dtype=torch.long)

        sample = {
            "z_t": z_t,
            "z_goal": z_goal,
            "steps_remaining": steps_remaining,
            "action": action,
        }
        if self.tokenizer is not None:
            tokens = self.tokenizer.actions_to_token_ids(action.numpy())
            sample["action_tokens"] = torch.from_numpy(tokens.astype(np.int64)).long()
        return sample
