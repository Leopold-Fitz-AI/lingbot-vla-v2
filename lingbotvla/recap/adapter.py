"""Parameter-efficient, null-preserving RECAP condition adapter helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_condition_ids(condition_ids, *, batch_size, device):
    if condition_ids is None:
        ids = torch.full((batch_size,), -1, device=device, dtype=torch.long)
    else:
        ids = torch.as_tensor(condition_ids, device=device, dtype=torch.long).reshape(-1)
    if ids.numel() != batch_size:
        raise ValueError(
            "condition_ids must contain one value per batch item, got "
            f"{ids.numel()} for batch size {batch_size}"
        )
    if not torch.all((ids >= -1) & (ids <= 1)):
        raise ValueError("condition_ids values must be -1, 0, or 1")
    return ids


def build_recap_condition_embedding(
    condition_embeddings: torch.Tensor,
    condition_ids,
    *,
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
    scale: float = 1.0,
) -> torch.Tensor:
    """Build condition embeddings for ids ``null=-1, negative=0, positive=1``.

    ``condition_embeddings`` stores only negative and positive rows. The null
    row is derived as an exact zero with a zero-gradient graph edge. This keeps
    null inference identical to the base policy and permits all-null batches
    when the adapter is the only trainable parameter.
    """

    ids = _normalize_condition_ids(
        condition_ids, batch_size=batch_size, device=device
    )
    if condition_embeddings.ndim != 2 or condition_embeddings.shape[0] != 2:
        raise ValueError("condition_embeddings must have shape [2, hidden_size]")

    null_embedding = condition_embeddings.sum(dim=0, keepdim=True) * 0.0
    table = torch.cat((null_embedding, condition_embeddings), dim=0)
    return F.embedding(ids + 1, table).to(dtype=dtype) * float(scale)


def apply_recap_velocity_lora(
    hidden_states: torch.Tensor,
    condition_ids,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    *,
    scale: float = 1.0,
    signed_axis: bool = False,
) -> torch.Tensor:
    """Return a condition-gated low-rank velocity residual.

    Args:
        hidden_states: Action-expert output with shape ``[B, T, H]``.
        condition_ids: One id per batch item (null=-1, negative=0, positive=1).
        lora_a: Condition-specific down projections with shape ``[2, R, H]``.
        lora_b: Condition-specific up projections with shape ``[2, D, R]``.

    The null residual is exactly zero, while retaining a valid zero-gradient
    graph edge for adapter-only training.
    """

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [B, T, H]")
    batch_size, _, hidden_size = hidden_states.shape
    if lora_a.ndim != 3 or lora_a.shape[0] != 2 or lora_a.shape[2] != hidden_size:
        raise ValueError("lora_a must have shape [2, rank, hidden_size]")
    if lora_b.ndim != 3 or lora_b.shape[0] != 2 or lora_b.shape[2] != lora_a.shape[1]:
        raise ValueError("lora_b must have shape [2, action_dim, rank]")

    ids = _normalize_condition_ids(
        condition_ids,
        batch_size=batch_size,
        device=hidden_states.device,
    )
    # Null indexes a row temporarily, then gets masked to an exact zero. A
    # signed axis derives negative as the exact inverse of the learned positive
    # residual; this avoids fitting a second BC policy to confounded failures.
    row_indices = (
        torch.ones_like(ids)
        if signed_axis
        else ids.clamp_min(0)
    )
    selected_a = lora_a[row_indices].to(dtype=hidden_states.dtype)
    selected_b = lora_b[row_indices].to(dtype=hidden_states.dtype)
    low_rank = torch.einsum("bth,brh->btr", hidden_states, selected_a)
    residual = torch.einsum("btr,bdr->btd", low_rank, selected_b)
    if signed_axis:
        active = torch.where(ids == 0, -1, ids.clamp_min(0)).to(
            dtype=residual.dtype
        )
    else:
        active = (ids != -1).to(dtype=residual.dtype)
    rank_scale = float(scale) / float(max(1, lora_a.shape[1]))
    return residual * active.view(batch_size, 1, 1) * rank_scale
