from __future__ import annotations

import torch
from torch import nn

from jepa import JEPA


class JEPAWithIDM(JEPA):
    """Original JEPA plus an optional inverse dynamics decoder."""

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        projector: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
        inverse_decoder: nn.Module | None = None,
    ):
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj,
        )
        self.inverse_decoder = inverse_decoder

    def inverse_decode(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        if self.inverse_decoder is None:
            raise RuntimeError("inverse_decoder is not configured")
        return self.inverse_decoder(z_t, z_next)

