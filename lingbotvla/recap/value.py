"""Distributional value-learning building blocks for RECAP.

The original RECAP recipe models Monte-Carlo returns with 201 categorical
value bins.  This module deliberately stays independent of a particular vision
encoder: callers can attach :class:`CategoricalValueHead` to any pooled VLM
representation while sharing the return and loss implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CategoricalValueHead(nn.Module):
    """Map context embeddings to a categorical value distribution."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_bins: int = 201,
        value_min: float = -2000.0,
        value_max: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_bins < 2:
            raise ValueError(f"num_bins must be at least 2, got {num_bins}")
        if value_max <= value_min:
            raise ValueError(f"value_max must be greater than value_min, got {value_min}..{value_max}")
        self.projection = nn.Linear(hidden_size, num_bins)
        self.register_buffer(
            "value_support",
            torch.linspace(float(value_min), float(value_max), num_bins),
            persistent=True,
        )

    @property
    def num_bins(self) -> int:
        return int(self.value_support.numel())

    def forward(self, context_embedding: Tensor) -> Tensor:
        return self.projection(context_embedding)

    def probabilities(self, context_embedding: Tensor) -> Tensor:
        return F.softmax(self(context_embedding), dim=-1)

    def expected_value_from_logits(self, logits: Tensor) -> Tensor:
        if logits.shape[-1] != self.num_bins:
            raise ValueError(f"Expected {self.num_bins} value logits, got shape {tuple(logits.shape)}")
        support = self.value_support.to(device=logits.device, dtype=logits.dtype)
        return (F.softmax(logits, dim=-1) * support).sum(dim=-1)

    def expected_value(self, context_embedding: Tensor) -> Tensor:
        return self.expected_value_from_logits(self(context_embedding))


class StateTaskValueModel(nn.Module):
    """Small independent RECAP baseline conditioned on state and task ID.

    This model makes the first rollout/value/labeling loop runnable before a
    shared visual-language value encoder is available. It must be treated as a
    baseline: tasks whose progress is not observable from proprioception need a
    visual encoder in later RECAP iterations.
    """

    def __init__(
        self,
        state_dim: int,
        num_tasks: int,
        *,
        hidden_size: int = 256,
        task_embedding_dim: int = 64,
        num_bins: int = 201,
        value_min: float = -2000.0,
        value_max: float = 0.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive, got {num_tasks}")
        self.state_dim = int(state_dim)
        self.num_tasks = int(num_tasks)
        self.task_embedding = nn.Embedding(num_tasks, task_embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + task_embedding_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.value_head = CategoricalValueHead(
            hidden_size,
            num_bins=num_bins,
            value_min=value_min,
            value_max=value_max,
        )

    @property
    def value_support(self) -> Tensor:
        return self.value_head.value_support

    def forward(self, state: Tensor, task_id: Tensor) -> Tensor:
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, got {state.shape[-1]}")
        task_embedding = self.task_embedding(task_id.long())
        context = self.encoder(torch.cat((state, task_embedding), dim=-1))
        return self.value_head(context)

    def expected_value(self, state: Tensor, task_id: Tensor) -> Tensor:
        return self.value_head.expected_value_from_logits(self(state, task_id))


def discretize_returns(returns: Tensor, value_support: Tensor) -> Tensor:
    """Map scalar returns to the nearest categorical support index."""

    if value_support.ndim != 1 or value_support.numel() < 2:
        raise ValueError("value_support must be a one-dimensional tensor with at least two bins")
    if not torch.all(value_support[1:] > value_support[:-1]):
        raise ValueError("value_support must be strictly increasing")
    if not torch.isfinite(returns).all():
        raise ValueError("returns contains NaN or infinity")

    support = value_support.to(device=returns.device, dtype=returns.dtype)
    insertion = torch.searchsorted(support, returns.contiguous())
    upper = insertion.clamp(max=support.numel() - 1)
    lower = (upper - 1).clamp(min=0)
    lower_distance = (returns - support[lower]).abs()
    upper_distance = (support[upper] - returns).abs()
    return torch.where(upper_distance < lower_distance, upper, lower).long()


def categorical_value_loss(
    logits: Tensor,
    returns: Tensor,
    value_support: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Cross-entropy loss for discretized Monte-Carlo return targets."""

    if logits.shape[:-1] != returns.shape:
        raise ValueError(
            "Value logits and return targets must share leading dimensions, got "
            f"{tuple(logits.shape)} and {tuple(returns.shape)}"
        )
    targets = discretize_returns(returns, value_support)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(returns)
    if valid_mask is None:
        return losses.mean()
    if valid_mask.shape != returns.shape:
        raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} does not match returns {tuple(returns.shape)}")
    mask = valid_mask.to(device=losses.device, dtype=losses.dtype)
    return (losses * mask).sum() / mask.sum().clamp(min=1.0)


def monte_carlo_returns(
    rewards: Tensor,
    *,
    terminated: Tensor | None = None,
    valid_mask: Tensor | None = None,
    gamma: float = 1.0,
    bootstrap_value: Tensor | float | None = None,
) -> Tensor:
    """Compute masked Monte-Carlo returns along the last dimension.

    Each leading index is treated as an independent trajectory. ``terminated``
    cuts bootstrapping after true environment terminals. Invalid padded steps
    receive zero and do not propagate a return into earlier episodes.
    """

    if rewards.ndim == 0:
        raise ValueError("rewards must have a time dimension")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards contains NaN or infinity")

    if terminated is None:
        terminated = torch.zeros_like(rewards, dtype=torch.bool)
    if valid_mask is None:
        valid_mask = torch.ones_like(rewards, dtype=torch.bool)
    if terminated.shape != rewards.shape or valid_mask.shape != rewards.shape:
        raise ValueError("terminated and valid_mask must match rewards shape")

    if bootstrap_value is None:
        running = torch.zeros_like(rewards[..., 0])
    else:
        running = torch.as_tensor(
            bootstrap_value,
            device=rewards.device,
            dtype=rewards.dtype,
        )
        running = torch.broadcast_to(running, rewards.shape[:-1]).clone()

    output = torch.zeros_like(rewards)
    discount = float(gamma)
    for step in range(rewards.shape[-1] - 1, -1, -1):
        valid = valid_mask[..., step]
        continuation = (~terminated[..., step]).to(rewards.dtype)
        candidate = rewards[..., step] + discount * continuation * running
        running = torch.where(valid, candidate, torch.zeros_like(candidate))
        output[..., step] = torch.where(valid, running, torch.zeros_like(running))
    return output
