"""Classifier-free guidance utilities for RECAP inference."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def duplicate_cfg_denoise_inputs(
    state: Tensor,
    x_t: Tensor,
    timestep: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create a ``[positive, null]`` denoising batch with shared inputs."""

    if state.shape[0] != x_t.shape[0]:
        raise ValueError("state and x_t must have the same batch size")
    batch_size = state.shape[0]
    if timestep.numel() == 1:
        timestep = timestep.expand(batch_size)
    if timestep.shape != (batch_size,):
        raise ValueError(f"timestep must be scalar or [B], got {tuple(timestep.shape)}")
    return (
        torch.cat((state, state), dim=0),
        torch.cat((x_t, x_t), dim=0),
        torch.cat((timestep, timestep), dim=0),
    )


def combine_cfg_velocities(
    positive_velocity: Tensor,
    null_velocity: Tensor,
    scale: float,
) -> Tensor:
    """Apply advantage-only CFG to flow velocity predictions.

    A scale of zero selects the unconditional/null branch, while a scale of one
    exactly selects the positive branch.
    """

    if positive_velocity.shape != null_velocity.shape:
        raise ValueError(
            "positive and null velocities must have identical shapes, got "
            f"{tuple(positive_velocity.shape)} and {tuple(null_velocity.shape)}"
        )
    scale = float(scale)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError(f"CFG scale must be finite and non-negative, got {scale}")
    return null_velocity + scale * (positive_velocity - null_velocity)


def combine_cfg_velocity_batch(
    velocity: Tensor,
    batch_size: int,
    scale: float,
) -> Tensor:
    """Split a ``[positive, null]`` velocity batch and apply CFG."""

    if batch_size <= 0 or velocity.shape[0] != 2 * batch_size:
        raise ValueError(f"Expected a doubled velocity batch of size {2 * batch_size}, got {velocity.shape[0]}")
    positive_velocity, null_velocity = velocity.split(batch_size, dim=0)
    return combine_cfg_velocities(positive_velocity, null_velocity, scale)
