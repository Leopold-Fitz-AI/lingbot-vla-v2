"""Reward construction and advantage labeling for RECAP rollouts."""

from __future__ import annotations

from enum import IntEnum

import torch
from torch import Tensor


class RecapAdvantageLabel(IntEnum):
    """Numeric labels stored in rollout datasets."""

    NEUTRAL = -1
    NEGATIVE = 0
    POSITIVE = 1


def time_to_success_rewards(
    episode_length: int,
    *,
    success: bool,
    failure_penalty: float,
    device: torch.device | str | None = None,
) -> Tensor:
    """Construct the sparse time-to-success reward used by RECAP.

    Non-terminal decisions receive ``-1``. A successful terminal receives zero;
    a failed terminal receives ``-failure_penalty``.
    """

    if episode_length <= 0:
        raise ValueError(f"episode_length must be positive, got {episode_length}")
    if failure_penalty <= 0:
        raise ValueError(f"failure_penalty must be positive, got {failure_penalty}")
    rewards = torch.full((episode_length,), -1.0, device=device)
    rewards[-1] = 0.0 if success else -float(failure_penalty)
    return rewards


def expected_value_from_logits(logits: Tensor, value_support: Tensor) -> Tensor:
    """Convert categorical value logits to their scalar expectation."""

    if logits.shape[-1] != value_support.numel():
        raise ValueError(f"Value logits have {logits.shape[-1]} bins but support has {value_support.numel()}")
    support = value_support.to(device=logits.device, dtype=logits.dtype)
    return (torch.softmax(logits, dim=-1) * support).sum(dim=-1)


def compute_advantages(returns: Tensor, values: Tensor) -> Tensor:
    if returns.shape != values.shape:
        raise ValueError(
            f"returns and values must have the same shape, got {tuple(returns.shape)} and {tuple(values.shape)}"
        )
    if not torch.isfinite(returns).all() or not torch.isfinite(values).all():
        raise ValueError("returns and values must be finite")
    return returns - values


def label_advantages(
    advantages: Tensor,
    *,
    threshold: float = 0.0,
    neutral_margin: float = 0.0,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Binarize advantages, optionally reserving a neutral margin.

    Positive means ``advantage > threshold + neutral_margin`` and negative
    means ``advantage < threshold - neutral_margin``. Borderline or invalid
    values are labeled ``NEUTRAL`` and should not be used as a conditioned
    policy target.
    """

    if neutral_margin < 0:
        raise ValueError(f"neutral_margin must be non-negative, got {neutral_margin}")
    if not torch.isfinite(advantages).all():
        raise ValueError("advantages contains NaN or infinity")
    if valid_mask is not None and valid_mask.shape != advantages.shape:
        raise ValueError("valid_mask must match advantages shape")

    labels = torch.full_like(
        advantages,
        fill_value=int(RecapAdvantageLabel.NEUTRAL),
        dtype=torch.int8,
    )
    labels = torch.where(
        advantages > float(threshold) + float(neutral_margin),
        torch.ones_like(labels),
        labels,
    )
    labels = torch.where(
        advantages < float(threshold) - float(neutral_margin),
        torch.zeros_like(labels),
        labels,
    )
    if valid_mask is not None:
        labels = torch.where(
            valid_mask.to(device=labels.device, dtype=torch.bool),
            labels,
            torch.full_like(labels, int(RecapAdvantageLabel.NEUTRAL)),
        )
    return labels
