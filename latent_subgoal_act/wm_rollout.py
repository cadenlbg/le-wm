from __future__ import annotations

import torch


def _expected_action_dim(wm_model) -> int | None:
    action_encoder = getattr(wm_model, "action_encoder", None)
    patch_embed = getattr(action_encoder, "patch_embed", None)
    if patch_embed is not None and hasattr(patch_embed, "in_channels"):
        return int(patch_embed.in_channels)
    return None


def _to_wm_action_steps(wm_model, action_chunk: torch.Tensor) -> torch.Tensor:
    expected_dim = _expected_action_dim(wm_model)
    if expected_dim is None or expected_dim == action_chunk.shape[-1]:
        return action_chunk

    batch_size, horizon, action_dim = action_chunk.shape
    flattened_dim = horizon * action_dim
    if expected_dim == flattened_dim:
        return action_chunk.reshape(batch_size, 1, flattened_dim)

    if expected_dim % action_dim == 0:
        block = expected_dim // action_dim
        usable_horizon = (horizon // block) * block
        if usable_horizon <= 0:
            raise ValueError(
                f"LeWM action encoder expects {expected_dim} dims, but action chunk has {horizon}x{action_dim}."
            )
        action_chunk = action_chunk[:, :usable_horizon]
        return action_chunk.reshape(batch_size, usable_horizon // block, expected_dim)

    raise ValueError(
        f"Cannot adapt action chunk shape {tuple(action_chunk.shape)} to LeWM action dim {expected_dim}."
    )


def rollout_latent_with_actions(wm_model, z_t: torch.Tensor, action_chunk: torch.Tensor, history_size: int = 1) -> torch.Tensor:
    """Roll a frozen LeWM predictor in latent space.

    Args:
        wm_model: LeWM/JEPA-style model exposing action_encoder and predict.
        z_t: current latent, shape [B, D].
        action_chunk: normalized action chunk, shape [B, K, A]. If LeWM expects
            frameskip-flattened actions, this is reshaped to [B, 1, K*A].
        history_size: number of recent latent/action steps passed to the predictor.

    Returns:
        Predicted terminal latent after the full action chunk, shape [B, D].
    """

    if not hasattr(wm_model, "action_encoder") or not hasattr(wm_model, "predict"):
        raise AttributeError("wm_model must expose action_encoder and predict for latent rollout")

    wm_actions = _to_wm_action_steps(wm_model, action_chunk)
    history_size = max(1, int(history_size))
    emb = z_t.unsqueeze(1)
    actions_so_far = wm_actions[:, :0]

    for step in range(wm_actions.shape[1]):
        actions_so_far = torch.cat([actions_so_far, wm_actions[:, step : step + 1]], dim=1)
        act_emb = wm_model.action_encoder(actions_so_far)
        emb_hist = emb[:, -history_size:]
        act_hist = act_emb[:, -emb_hist.shape[1] :]
        pred_next = wm_model.predict(emb_hist, act_hist)[:, -1:]
        emb = torch.cat([emb, pred_next], dim=1)

    return emb[:, -1]
