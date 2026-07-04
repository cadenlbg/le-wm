from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset


class LatentSubgoalACTDataset(Dataset):
    def __init__(self, payload: Dict[str, Any], max_samples: Optional[int] = None):
        required = ("z_t", "z_g", "z_h", "action", "episode", "step", "goal_step", "subgoal_step")
        for key in required:
            if key not in payload:
                raise KeyError(f"missing required key in latent subgoal ACT payload: {key}")

        self.payload = payload
        self.length = int(payload["z_t"].shape[0])
        if max_samples is not None:
            self.length = min(self.length, int(max_samples))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "z_t": self.payload["z_t"][idx],
            "z_g": self.payload["z_g"][idx],
            "z_h": self.payload["z_h"][idx],
            "action": self.payload["action"][idx],
            "episode": self.payload["episode"][idx],
            "step": self.payload["step"][idx],
            "goal_step": self.payload["goal_step"][idx],
            "subgoal_step": self.payload["subgoal_step"][idx],
        }

