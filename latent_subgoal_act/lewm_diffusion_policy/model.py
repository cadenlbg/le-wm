from __future__ import annotations

from copy import deepcopy
import math
from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class LinearActionNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        self.mean = mean.float()
        self.std = std.float().clamp_min(eps)

    @classmethod
    def fit(cls, action: torch.Tensor, enabled: bool = True):
        if not enabled:
            mean = torch.zeros(action.shape[-1], dtype=torch.float32)
            std = torch.ones(action.shape[-1], dtype=torch.float32)
            return cls(mean, std)
        flat = action.reshape(-1, action.shape[-1]).float()
        return cls(flat.mean(dim=0), flat.std(dim=0, unbiased=False))

    @classmethod
    def load(cls, state):
        return cls(torch.as_tensor(state["mean"]).float(), torch.as_tensor(state["std"]).float())

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.mean.view(1, 1, -1)) / self.std.view(1, 1, -1)

    def unnormalize(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.std.view(1, 1, -1) + self.mean.view(1, 1, -1)

    def state_dict(self):
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu()}


class EMAModel:
    """EMA schedule copied in spirit from Diffusion Policy."""

    def __init__(self, model: nn.Module, update_after_step=0, inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999):
        self.averaged_model = deepcopy(model).eval().requires_grad_(False)
        self.update_after_step = int(update_after_step)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.optimization_step = 0
        self.decay = 0.0

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, int(optimization_step) - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, model: nn.Module):
        self.decay = self.get_decay(self.optimization_step)
        for ema_param, param in zip(self.averaged_model.parameters(), model.parameters()):
            if not param.requires_grad:
                ema_param.copy_(param.data)
            else:
                ema_param.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
        for ema_buffer, buffer in zip(self.averaged_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)
        self.optimization_step += 1

    def to(self, device):
        self.averaged_model.to(device)
        return self


def sinusoidal_pos_emb(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1))
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish, matching Diffusion Policy's block order."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(min(n_groups, out_channels), out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 5, n_groups: int = 8, cond_predict_scale: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.cond_predict_scale = bool(cond_predict_scale)
        self.out_channels = int(out_channels)
        cond_channels = 2 * out_channels if self.cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0]
            bias = embed[:, 1]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """Official-DP-style ConditionalUnet1D with robust horizon alignment."""

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: Optional[int],
        diffusion_step_embed_dim: int = 128,
        down_dims: Sequence[int] = (512, 1024, 2048),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        all_dims = [int(input_dim), *[int(x) for x in down_dims]]
        start_dim = all_dims[1]
        dsed = int(diffusion_step_embed_dim)
        self.diffusion_step_encoder = nn.Sequential(
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        self.diffusion_step_embed_dim = dsed
        cond_dim = dsed + (0 if global_cond_dim is None else int(global_cond_dim))
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        self.down_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                        ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
            ]
        )

        self.up_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = idx >= len(in_out) - 2
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale),
                        ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size), nn.Conv1d(start_dim, input_dim, 1))

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor | int, global_cond: Optional[torch.Tensor] = None):
        # sample: [B,H,A] -> [B,A,H]
        x = sample.transpose(1, 2)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(sinusoidal_pos_emb(timestep, self.diffusion_step_embed_dim))
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            skip = h.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat((x, skip), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        if x.shape[-1] != sample.shape[1]:
            x = F.interpolate(x, size=sample.shape[1], mode="nearest")
        return x.transpose(1, 2)


def cosine_beta_schedule(num_steps: int, s: float = 0.008, device="cpu"):
    x = torch.linspace(0, int(num_steps), int(num_steps) + 1, device=device)
    alpha_bar = torch.cos(((x / int(num_steps)) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-4, 0.999)


class DDPMScheduler:
    def __init__(self, num_train_timesteps: int = 100, beta_schedule: str = "squaredcos_cap_v2", beta_start: float = 1e-4, beta_end: float = 2e-2, clip_sample: bool = True):
        self.num_train_timesteps = int(num_train_timesteps)
        self.beta_schedule = beta_schedule
        if beta_schedule in ("squaredcos_cap_v2", "cosine"):
            betas = cosine_beta_schedule(self.num_train_timesteps)
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule={beta_schedule}")
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.clip_sample = bool(clip_sample)
        self.timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.timesteps = self.timesteps.to(device)
        return self

    def state_config(self):
        return {
            "num_train_timesteps": self.num_train_timesteps,
            "beta_schedule": self.beta_schedule,
            "clip_sample": self.clip_sample,
        }

    def set_timesteps(self, num_inference_steps: int):
        # simple evenly spaced DDPM subset, reversed.
        steps = torch.linspace(0, self.num_train_timesteps - 1, int(num_inference_steps), device=self.betas.device).long()
        self.timesteps = torch.flip(steps, dims=[0])

    def add_noise(self, samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        return alpha_bar.sqrt() * samples + (1.0 - alpha_bar).sqrt() * noise

    def step(self, model_output: torch.Tensor, timestep: int | torch.Tensor, sample: torch.Tensor, generator=None):
        t = int(timestep.item() if torch.is_tensor(timestep) else timestep)
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]
        pred_original = (sample - (1 - alpha_bar).sqrt() * model_output) / alpha_bar.sqrt()
        if self.clip_sample:
            pred_original = pred_original.clamp(-1.0, 1.0)
        prev = (sample - beta / (1.0 - alpha_bar).sqrt() * model_output) / alpha.sqrt()
        if t > 0:
            prev = prev + beta.sqrt() * torch.randn(sample.shape, device=sample.device, dtype=sample.dtype, generator=generator)
        return prev, pred_original


class LeWMLatentDiffusionPolicy(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        horizon: int,
        n_action_steps: int = 5,
        history_size: int = 2,
        goal_condition: bool = True,
        num_inference_steps: int = 100,
        diffusion_step_embed_dim: int = 128,
        down_dims: Sequence[int] = (512, 1024, 2048),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.n_action_steps = int(n_action_steps)
        self.history_size = int(history_size)
        self.goal_condition = bool(goal_condition)
        self.num_inference_steps = int(num_inference_steps)
        global_cond_dim = (self.history_size + (1 if self.goal_condition else 0)) * self.latent_dim
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

    def make_global_cond(self, z_history: torch.Tensor, z_g: Optional[torch.Tensor] = None):
        if z_history.ndim == 2:
            z_history = z_history.unsqueeze(1).expand(-1, self.history_size, -1)
        z_history = z_history[:, -self.history_size :]
        if z_history.shape[1] < self.history_size:
            pad = z_history[:, :1].expand(-1, self.history_size - z_history.shape[1], -1)
            z_history = torch.cat([pad, z_history], dim=1)
        pieces = [z_history.reshape(z_history.shape[0], -1)]
        if self.goal_condition:
            if z_g is None:
                raise ValueError("z_g is required when goal_condition=True")
            pieces.append(z_g)
        return torch.cat(pieces, dim=-1)

    def compute_loss(self, batch, scheduler: DDPMScheduler, normalizer: LinearActionNormalizer):
        action = normalizer.normalize(batch["action"])
        batch_size = action.shape[0]
        noise = torch.randn_like(action)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=action.device).long()
        noisy_action = scheduler.add_noise(action, noise, timesteps)
        z_history = batch.get("z_history", batch["z_t"])
        global_cond = self.make_global_cond(z_history, batch.get("z_g"))
        pred = self.model(noisy_action, timesteps, global_cond=global_cond)
        loss = F.mse_loss(pred, noise, reduction="none")
        return loss.reshape(batch_size, -1).mean(dim=1).mean()

    @torch.no_grad()
    def conditional_sample(self, z_history, z_g, scheduler: DDPMScheduler, num_samples: int = 1, generator=None):
        if z_history.ndim == 2:
            z_history = z_history.unsqueeze(1).expand(-1, self.history_size, -1)
        batch_size = z_history.shape[0]
        num_samples = int(num_samples)
        trajectory = torch.randn(batch_size * num_samples, self.horizon, self.action_dim, device=z_history.device, generator=generator)
        flat_z_history = z_history.unsqueeze(1).expand(batch_size, num_samples, -1, -1).reshape(batch_size * num_samples, z_history.shape[1], -1)
        flat_z_g = z_g.unsqueeze(1).expand(batch_size, num_samples, -1).reshape(batch_size * num_samples, -1) if z_g is not None else None
        global_cond = self.make_global_cond(flat_z_history, flat_z_g)
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            pred = self.model(trajectory, t, global_cond=global_cond)
            trajectory, _ = scheduler.step(pred, t, trajectory, generator=generator)
        return trajectory.reshape(batch_size, num_samples, self.horizon, self.action_dim)
