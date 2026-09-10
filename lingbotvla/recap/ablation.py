"""Deterministic schedules and guarded compact optimization for the P0 study.

These seeds identify optimizer/noise draws, NOT new simulator environment seeds.
The four cells share data, times, noise and the same single-device execution frame.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


TASKS = {
    "hanging_mug": ("signed", 26, 5),
    "place_can_basket": ("regularized", 40, 6),
    "stack_bowls_three": ("positive", 18, 7),
}
SEEDS = (2101, 2102, 2103)
INITIALIZATIONS = ("legacy_sin_v1", "orthogonal_matched_v1")
BACKENDS = ("legacy", "deployment")
STEPS, BATCH = 50, 4
INITIALIZATION_SEED = 971


def sample_schedule(rows, seed, steps=STEPS, batch=BATCH):
    """Shuffled full epochs, drop_last=True, exactly steps*batch presentations."""
    if rows < batch or steps <= 0 or batch <= 0:
        raise ValueError("Invalid complete-batch schedule")
    rng = np.random.Generator(np.random.PCG64(seed))
    result = []
    while len(result) < steps:
        order = rng.permutation(rows)[: rows // batch * batch]
        result.extend(order.reshape(-1, batch).tolist())
    return result[:steps]


def draw_seed(task, seed, step, micro):
    payload = json.dumps(["recap-p0-draw-v1", task, seed, step, micro], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little") % (2**63 - 1)


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def assert_training_sources(paths):
    for path in paths:
        parts = Path(path).parts
        if "final" in parts or any(p in {"s50", "s51", "s52"} for p in parts):
            raise ValueError("Final data cannot enter the repair training study")


def cached_losses(hidden, base, actions, noise, condition, a, b, joint_mask, action_is_pad, *, scale, weight):
    """Same masked L1 FM + residual L2 as the real policy wrapper; not DPO."""
    import torch

    from lingbotvla.recap.adapter import apply_recap_velocity_lora
    from lingbotvla.recap.loss import masked_action_loss

    if hidden.requires_grad or base.requires_grad:
        raise ValueError("Only detached frozen features can be cached")
    residual = apply_recap_velocity_lora(hidden, condition, a, b, scale=scale, signed_axis=True)
    error = (noise - actions - (base + residual)).abs()
    kwargs = dict(action_dim=actions.shape[-1], joint_mask=joint_mask, action_is_pad=action_is_pad)
    fm, _ = masked_action_loss(error, **kwargs)
    penalty, _ = masked_action_loss(residual.square(), **kwargs)
    total = fm + weight * penalty
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite cached flow loss")
    return total, fm, penalty


def checked_gradients(parameters, *, step):
    import torch

    if len(parameters) != 2:
        raise ValueError("Expected A and B only")
    norms = []
    for p in parameters:
        if p.grad is None or not torch.isfinite(p.grad).all() or not torch.isfinite(p).all():
            raise FloatingPointError("Missing/nonfinite adapter gradient or parameter")
        norms.append(float(torch.linalg.vector_norm(p.grad)))
    if step == 0 and (norms[0] != 0 or norms[1] == 0):
        raise ValueError("First-step gradient gate failed: expected zero A and nonzero B")
    if step == 1 and any(n == 0 for n in norms):
        raise ValueError("Second-step gradient gate failed: A and B must both receive gradients")
    return norms


def summarize_pairs(rows):
    """Paired flow-error surrogate, descriptive only on already consumed states."""
    groups = {}
    for row in rows:
        key = (row["split"], row["seed"], row["pair_id"], row["time"], row["noise_seed"])
        pair = groups.setdefault(key, {})
        label = row["label"]
        if label not in (0, 1) or label in pair:
            raise ValueError("Duplicate/invalid pair member")
        pair[label] = row
    if not groups or any(set(pair) != {0, 1} for pair in groups.values()):
        raise ValueError("Incomplete diagnostic pairs")
    result = {}
    for split in sorted({key[0] for key in groups}):
        members = [(key, pair) for key, pair in groups.items() if key[0] == split]
        margin = [
            (pair[1]["base_fm"] - pair[1]["positive_fm"]) - (pair[0]["base_fm"] - pair[0]["positive_fm"])
            for _, pair in members
        ]
        result[split] = {
            "states": len({key[1] for key, _ in members}),
            "paired_draws": len(members),
            "surrogate_margin_mean": float(np.mean(margin)),
            "positive_margin_fraction": float(np.mean(np.array(margin) > 0)),
            "positive_bc_improvement_mean": float(
                np.mean([pair[1]["base_fm"] - pair[1]["positive_fm"] for _, pair in members])
            ),
            "scope": "Consumed-state flow-error surrogate; not independent policy success",
        }
    return result
