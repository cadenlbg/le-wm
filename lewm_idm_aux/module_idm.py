from __future__ import annotations

import torch
from torch import nn


class InverseDynamicsDecoder(nn.Module):
    """Predict raw action blocks from adjacent LeWM latents.

    The decoder is deliberately small and LGP-style: a stack of MLP blocks over
    concat(z_t, z_next). It predicts embedding-before action blocks, not action
    embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        action_dim: int = 10,
        hidden_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim

        act_cls = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
        layers: list[nn.Module] = []
        in_dim = 2 * embed_dim
        for _ in range(n_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    act_cls(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, action_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """Return raw action-block predictions.

        Args:
            z_t: (..., embed_dim)
            z_next: (..., embed_dim)

        Returns:
            (..., action_dim)
        """
        h = torch.cat([z_t, z_next], dim=-1)
        return self.head(self.backbone(h))

