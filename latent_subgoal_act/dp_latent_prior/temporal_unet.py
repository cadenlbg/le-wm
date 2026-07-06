from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, emb.new_zeros(emb.shape[0], 1)], dim=1)
    return emb


class FiLMResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        padding = kernel_size // 2
        groups1 = min(n_groups, in_channels)
        groups2 = min(n_groups, out_channels)
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups1, in_channels),
            nn.Mish(),
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
        )
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * out_channels))
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups2, out_channels),
            nn.Mish(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
        )
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        scale, bias = self.film(cond).chunk(2, dim=-1)
        h = h * (1.0 + scale.unsqueeze(-1)) + bias.unsqueeze(-1)
        h = self.block2(h)
        return h + self.residual(x)


class Downsample1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, target_len: int):
        x = self.conv(x)
        if x.shape[-1] > target_len:
            x = x[..., :target_len]
        elif x.shape[-1] < target_len:
            x = torch.nn.functional.pad(x, (0, target_len - x.shape[-1]))
        return x


class LatentTemporalUnet(nn.Module):
    """Diffusion-Policy-style 1D U-Net for action chunks conditioned on z_t,z_g."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        prediction_horizon: int,
        down_dims: Sequence[int] = (256, 512, 1024),
        diffusion_step_embed_dim: int = 128,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.prediction_horizon = int(prediction_horizon)
        self.diffusion_step_embed_dim = int(diffusion_step_embed_dim)
        cond_dim = self.diffusion_step_embed_dim + 2 * self.latent_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 4 * self.diffusion_step_embed_dim),
            nn.Mish(),
            nn.Linear(4 * self.diffusion_step_embed_dim, 4 * self.diffusion_step_embed_dim),
        )
        cond_out = 4 * self.diffusion_step_embed_dim

        dims = [self.action_dim, *[int(d) for d in down_dims]]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for idx in range(len(dims) - 1):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        FiLMResidualBlock1D(dims[idx], dims[idx + 1], cond_out, kernel_size, n_groups),
                        FiLMResidualBlock1D(dims[idx + 1], dims[idx + 1], cond_out, kernel_size, n_groups),
                    ]
                )
            )
            self.downsamples.append(Downsample1D(dims[idx + 1]) if idx < len(dims) - 2 else nn.Identity())

        mid_dim = dims[-1]
        self.mid_blocks = nn.ModuleList(
            [
                FiLMResidualBlock1D(mid_dim, mid_dim, cond_out, kernel_size, n_groups),
                FiLMResidualBlock1D(mid_dim, mid_dim, cond_out, kernel_size, n_groups),
            ]
        )

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        rev_dims = list(reversed(dims[1:]))
        for idx in range(len(rev_dims) - 1):
            in_dim = rev_dims[idx] + rev_dims[idx + 1]
            out_dim = rev_dims[idx + 1]
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        FiLMResidualBlock1D(in_dim, out_dim, cond_out, kernel_size, n_groups),
                        FiLMResidualBlock1D(out_dim, out_dim, cond_out, kernel_size, n_groups),
                    ]
                )
            )
            self.upsamples.append(Upsample1D(out_dim) if idx < len(rev_dims) - 2 else nn.Identity())

        self.final = nn.Sequential(
            nn.GroupNorm(min(n_groups, dims[1]), dims[1]),
            nn.Mish(),
            nn.Conv1d(dims[1], self.action_dim, kernel_size=1),
        )

    def _condition(self, diffusion_step: torch.Tensor, z_t: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(diffusion_step, self.diffusion_step_embed_dim)
        return self.cond_mlp(torch.cat([t_emb, z_t, z_g], dim=-1))

    def forward(self, noisy_action: torch.Tensor, diffusion_step: torch.Tensor, z_t: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        leading_shape = noisy_action.shape[:-2]
        action = noisy_action.reshape(-1, self.prediction_horizon, self.action_dim).transpose(1, 2)
        z_t = z_t.reshape(-1, self.latent_dim)
        z_g = z_g.reshape(-1, self.latent_dim)
        diffusion_step = diffusion_step.reshape(-1)
        cond = self._condition(diffusion_step, z_t, z_g)

        skips = []
        h = action
        for blocks, downsample in zip(self.down_blocks, self.downsamples):
            h = blocks[0](h, cond)
            h = blocks[1](h, cond)
            skips.append(h)
            h = downsample(h)

        for block in self.mid_blocks:
            h = block(h, cond)

        skips = skips[:-1][::-1]
        for idx, blocks in enumerate(self.up_blocks):
            skip = skips[idx]
            if h.shape[-1] != skip.shape[-1]:
                h = torch.nn.functional.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = blocks[0](h, cond)
            h = blocks[1](h, cond)
            upsample = self.upsamples[idx]
            if isinstance(upsample, Upsample1D) and idx + 1 < len(skips):
                h = upsample(h, skips[idx + 1].shape[-1])
            elif isinstance(upsample, Upsample1D):
                h = upsample(h, self.prediction_horizon)

        out = self.final(h)
        if out.shape[-1] != self.prediction_horizon:
            out = torch.nn.functional.interpolate(out, size=self.prediction_horizon, mode="nearest")
        return out.transpose(1, 2).reshape(*leading_shape, self.prediction_horizon, self.action_dim)

