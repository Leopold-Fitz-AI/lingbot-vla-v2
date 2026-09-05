"""Parameter-efficient, null-preserving RECAP condition adapter helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_counterfactual_decision_map(path: str | Path) -> dict:
    """Load a strict canonical-task to non-negative decision-index map."""

    map_path = Path(path).resolve()
    with map_path.open() as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("tasks"), dict):
        raise ValueError(
            "Counterfactual decision map must have schema_version=1 and a tasks object"
        )
    tasks = {}
    for task, decision in payload["tasks"].items():
        if not isinstance(task, str) or not task:
            raise ValueError("Counterfactual decision map has an invalid task name")
        if isinstance(decision, bool) or not isinstance(decision, int) or decision < 0:
            raise ValueError(
                f"Counterfactual decision for {task!r} must be a non-negative integer"
            )
        tasks[task] = decision
    return {**payload, "path": map_path, "tasks": tasks}


def load_recap_adapter_registry(path: str | Path) -> dict:
    """Load a task-to-adapter registry and verify every compact artifact.

    Artifact paths may be relative to the registry file. SHA-256 is mandatory
    so a corrupt or stale task adapter can never silently alter the policy.
    """

    registry_path = Path(path).resolve()
    with registry_path.open() as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("RECAP adapter registry schema_version must be 1")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("RECAP adapter registry must contain a tasks object")

    resolved = {}
    for task, entry in tasks.items():
        if not isinstance(task, str) or not task.strip():
            raise ValueError("RECAP adapter registry task names must be non-empty")
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"Registry entry for {task!r} must contain path")
        expected_sha256 = entry.get("sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(f"Registry entry for {task!r} requires SHA-256")
        artifact = Path(entry["path"])
        if not artifact.is_absolute():
            artifact = registry_path.parent / artifact
        artifact = artifact.resolve()
        if not artifact.is_file():
            raise FileNotFoundError(f"RECAP adapter for {task!r} not found: {artifact}")
        digest = hashlib.sha256()
        with artifact.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256.lower():
            raise ValueError(
                f"RECAP adapter SHA-256 mismatch for {task!r}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        resolved[task] = {
            **entry,
            "path": artifact,
            "sha256": actual_sha256,
        }
    return {
        **payload,
        "registry_path": registry_path,
        "tasks": resolved,
    }


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
