"""Loss helpers for RECAP action-chunk datasets."""

from __future__ import annotations

import torch


def masked_action_loss(
    losses: torch.Tensor,
    *,
    action_dim: int,
    joint_mask: torch.Tensor | None = None,
    action_is_pad: torch.Tensor | None = None,
    repeated_loss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce per-element action losses while excluding padded time steps."""
    if losses.ndim != 3:
        raise ValueError(f"losses must have shape [B,T,D], got {tuple(losses.shape)}")
    if joint_mask is not None:
        if repeated_loss:
            joint_mask = joint_mask.repeat(2, 1, 1)
        if joint_mask.shape != losses.shape:
            raise ValueError(
                "joint_mask must match action loss shape, got "
                f"{tuple(joint_mask.shape)} versus {tuple(losses.shape)}"
            )
        loss_mask = joint_mask.to(device=losses.device, dtype=torch.bool)
    else:
        losses = losses[:, :, :action_dim]
        loss_mask = torch.ones_like(losses, dtype=torch.bool)

    if action_is_pad is not None:
        if repeated_loss:
            action_is_pad = action_is_pad.repeat(2, 1)
        if action_is_pad.shape != losses.shape[:2]:
            raise ValueError(
                "action_is_pad must match action loss batch/time dimensions, got "
                f"{tuple(action_is_pad.shape)} versus {tuple(losses.shape[:2])}"
            )
        valid_time = ~action_is_pad.to(device=losses.device, dtype=torch.bool)
        loss_mask = loss_mask & valid_time.unsqueeze(-1)

    masked_losses = losses * loss_mask
    valid_counts = loss_mask.sum(dim=(1, 2)).clamp(min=1)
    batch_mean_losses = masked_losses.sum(dim=(1, 2)) / valid_counts
    loss = masked_losses.sum() / loss_mask.sum().clamp(min=1)
    return loss, batch_mean_losses
