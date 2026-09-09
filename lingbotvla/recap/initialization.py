"""Versioned velocity-LoRA initialization; no process-global RNG draws."""

from __future__ import annotations

import math

import torch


LEGACY_INITIALIZATION = "legacy_sin_v1"
ORTHOGONAL_INITIALIZATION = "orthogonal_matched_v1"
INITIALIZATIONS = (LEGACY_INITIALIZATION, ORTHOGONAL_INITIALIZATION)


@torch.no_grad()
def initialize_velocity_lora_(a, b, *, scheme=LEGACY_INITIALIZATION, seed=0, init_std=0.02):
    """Initialize A and zero B, preserving old artifacts and legacy arithmetic.

    The new scheme uses CPU float64 QR with a private generator. Each condition
    has the Frobenius norm of its legacy CPU-FP32 sine matrix, not a larger
    residual scale. Device/dtype conversion happens only after construction.
    Existing checkpoints overwrite these parameters normally.
    """
    if scheme not in INITIALIZATIONS:
        raise ValueError(f"Unknown RECAP initialization: {scheme}")
    if a.ndim != 3 or a.shape[0] != 2 or b.ndim != 3 or b.shape[0] != 2 or b.shape[2] != a.shape[1]:
        raise ValueError("Expected A [2,rank,hidden] and B [2,action_dim,rank]")
    if not a.is_floating_point() or not b.is_floating_point():
        raise ValueError("RECAP parameters must be floating point")
    if not math.isfinite(init_std) or init_std < 0:
        raise ValueError("recap_adapter_init_std must be finite and non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("recap_adapter_init_seed must be an integer in [0, 2**63)")
    rank, hidden = a.shape[1:]
    if rank <= 0 or hidden <= 0:
        raise ValueError("RECAP rank and hidden size must be positive")
    if scheme == LEGACY_INITIALIZATION:
        # Keep the exact original operation/device/dtype order for compatibility.
        values = torch.arange(a.numel(), device=a.device, dtype=torch.float32).reshape_as(a)
        a.copy_((torch.sin(values * 0.017) * init_std).to(dtype=a.dtype))
    else:
        if rank > hidden or init_std == 0:
            raise ValueError("Full-row-rank initialization requires rank <= hidden and init_std > 0")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        values = torch.arange(a.numel(), device="cpu", dtype=torch.float32).reshape(a.shape)
        reference = (torch.sin(values * 0.017) * init_std).double()
        rows = []
        for condition in range(2):
            q, r = torch.linalg.qr(torch.randn(hidden, rank, generator=generator, dtype=torch.float64))
            # Resolve QR sign ambiguity without consuming another RNG stream.
            q = q * torch.where(r.diag() < 0, -1.0, 1.0)
            rows.append(q.T * (torch.linalg.vector_norm(reference[condition]) / math.sqrt(rank)))
        a.copy_(torch.stack(rows).to(device=a.device, dtype=a.dtype))
    b.zero_()
    return {
        "scheme": scheme,
        "seed": seed,
        "init_std": init_std,
        "generator_device": "cpu" if scheme == ORTHOGONAL_INITIALIZATION else None,
        "generator_dtype": "float64" if scheme == ORTHOGONAL_INITIALIZATION else None,
        "norm_reference": "legacy_sin_v1_cpu_fp32" if scheme == ORTHOGONAL_INITIALIZATION else None,
        "target_device": str(a.device),
        "target_dtype": str(a.dtype),
        "a_shape": list(a.shape),
        "b_shape": list(b.shape),
    }
