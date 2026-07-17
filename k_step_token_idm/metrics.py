"""Losses, decoding helpers, and offline metrics for K-step token IDM."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from single_step_token_idm.tokenization import ActionTokenizer


def expected_actions_from_logits(logits: Tensor, tokenizer: ActionTokenizer) -> Tensor:
    """Decode categorical expectations for logits shaped ``(..., D, bins)``."""
    probabilities = torch.softmax(logits, dim=-1)
    centers = tokenizer.torch_bin_centers(logits.device, logits.dtype)
    normalized = torch.sum(probabilities * centers, dim=-1)
    action_dim = normalized.shape[-1]
    shape = [1] * (normalized.ndim - 1) + [action_dim]
    low = torch.as_tensor(tokenizer.action_low, device=logits.device, dtype=logits.dtype).view(shape)
    high = torch.as_tensor(tokenizer.action_high, device=logits.device, dtype=logits.dtype).view(shape)
    return 0.5 * (normalized + 1.0) * (high - low + tokenizer.eps) + low


def token_ids_to_actions(tokens: Tensor, tokenizer: ActionTokenizer) -> Tensor:
    """Decode token ids to continuous actions without leaving the current device."""
    bin_ids = (tokens.long() - tokenizer.token_offset).clamp(0, tokenizer.n_bins - 1)
    centers = tokenizer.torch_bin_centers(tokens.device, torch.float32)
    normalized = centers[bin_ids]
    action_dim = normalized.shape[-1]
    shape = [1] * (normalized.ndim - 1) + [action_dim]
    low = torch.as_tensor(tokenizer.action_low, device=tokens.device).view(shape)
    high = torch.as_tensor(tokenizer.action_high, device=tokens.device).view(shape)
    return 0.5 * (normalized + 1.0) * (high - low + tokenizer.eps) + low


def compute_training_loss(
    logits: Tensor,
    target_tokens: Tensor,
    target_actions: Tensor,
    tokenizer: ActionTokenizer,
    l1_coef: float,
) -> tuple[Tensor, dict[str, float]]:
    target_bins = target_tokens.long() - tokenizer.token_offset
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_bins.reshape(-1))
    expected_actions = expected_actions_from_logits(logits, tokenizer)
    l1 = F.l1_loss(expected_actions, target_actions)
    probabilities = torch.softmax(logits, dim=-1)
    entropy = -(probabilities * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    loss = ce + l1_coef * l1
    return loss, {
        "loss": float(loss.item()),
        "ce": float(ce.item()),
        "l1": float(l1.item()),
        "entropy": float(entropy.item()),
    }


@torch.no_grad()
def evaluate_offline(model, loader, tokenizer: ActionTokenizer, device: torch.device) -> dict[str, float]:
    """Compute teacher-forced and free-running metrics."""
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    per_step_l1 = None
    per_step_acc = None
    total_examples = 0

    for batch in loader:
        z_t = batch["z_t"].to(device)
        z_goal = batch["z_goal"].to(device)
        steps = batch["steps_remaining"].to(device)
        actions = batch["actions"].to(device)
        targets = batch["action_tokens"].to(device)
        logits = model(z_t, z_goal, steps, targets)
        teacher_actions = expected_actions_from_logits(logits, tokenizer)
        teacher_tokens = torch.argmax(logits, dim=-1) + tokenizer.token_offset
        generated = model.generate(z_t, z_goal, steps)[:, 0]
        generated_actions = token_ids_to_actions(generated, tokenizer)
        target_bins = targets.long() - tokenizer.token_offset

        batch_examples = actions.shape[0]
        totals["teacher_ce"] += batch_examples * float(
            F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_bins.reshape(-1)).item()
        )
        totals["teacher_l1"] += batch_examples * float(F.l1_loss(teacher_actions, actions).item())
        totals["teacher_token_acc"] += batch_examples * float((teacher_tokens == targets).float().mean().item())
        totals["free_l1"] += batch_examples * float(F.l1_loss(generated_actions, actions).item())
        totals["free_token_acc"] += batch_examples * float((generated == targets).float().mean().item())
        probabilities = torch.softmax(logits, dim=-1)
        totals["teacher_entropy"] += batch_examples * float(
            (-(probabilities * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()).item()
        )

        step_l1 = torch.abs(generated_actions - actions).mean(dim=(0, 2)).cpu().numpy()
        step_acc = (generated == targets).float().mean(dim=(0, 2)).cpu().numpy()
        if per_step_l1 is None:
            per_step_l1 = np.zeros_like(step_l1, dtype=np.float64)
            per_step_acc = np.zeros_like(step_acc, dtype=np.float64)
        per_step_l1 += step_l1 * batch_examples
        per_step_acc += step_acc * batch_examples
        total_examples += batch_examples

    if total_examples == 0:
        raise ValueError("offline evaluation loader is empty")
    result = {key: value / total_examples for key, value in totals.items()}
    for index, value in enumerate(per_step_l1 / total_examples):
        result[f"free_step_{index}_l1"] = float(value)
    for index, value in enumerate(per_step_acc / total_examples):
        result[f"free_step_{index}_token_acc"] = float(value)
    return result
