from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset


class LatentACTDataset(Dataset):
    def __init__(self, payload: Dict[str, Any], max_samples: Optional[int] = None):
        required = ("z_t", "z_g", "delta_z", "action")
        for key in required:
            if key not in payload:
                raise KeyError(f"missing required key in latent ACT payload: {key}")

        self.payload = payload
        self.length = int(payload["z_t"].shape[0])
        if max_samples is not None:
            self.length = min(self.length, int(max_samples))

        self.has_latent_target = "latent_target" in payload

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {
            "z_t": self.payload["z_t"][idx],
            "z_g": self.payload["z_g"][idx],
            "delta_z": self.payload["delta_z"][idx],
            "action": self.payload["action"][idx],
            "episode": self.payload["episode"][idx],
            "step": self.payload["step"][idx],
            "goal_step": self.payload["goal_step"][idx],
        }

        if self.has_latent_target:
            item["latent_target"] = self.payload["latent_target"][idx]
        else:
            item["latent_target"] = self.payload["z_g"][idx]
        return item

