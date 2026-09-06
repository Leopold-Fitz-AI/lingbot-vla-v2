"""Deterministic common-random-number schedules for causal rollouts."""

from __future__ import annotations

import hashlib


_MAX_TORCH_SEED = 2**63 - 1


def policy_action_seed(
    *,
    policy_seed: int,
    continuation_policy_seed: int | None,
    counterfactual_decision: int,
    decision_index: int,
    task_name: str | None = None,
    environment_seed: int | None = None,
    per_episode: bool = False,
) -> int | None:
    """Return the action-noise seed for one policy decision.

    With no continuation seed, the policy RNG is left untouched. Legacy mode
    reproduces the original one-intervention schedule. Per-episode mode hashes
    task and environment seed so conditions share random numbers while distinct
    episodes do not all reuse the same diffusion-noise path.
    """
    if continuation_policy_seed is None:
        return None
    if decision_index < 0 or counterfactual_decision < 0:
        raise ValueError("decision indices must be non-negative")
    branch = decision_index == counterfactual_decision
    source_seed = int(policy_seed if branch else continuation_policy_seed)
    if not per_episode:
        return source_seed if branch else source_seed + decision_index
    if not task_name:
        raise ValueError("per-episode common noise requires a canonical task name")
    if environment_seed is None:
        raise ValueError("per-episode common noise requires an environment seed")
    payload = (
        "lingbot-recap-common-noise-v1\0"
        f"{source_seed}\0{task_name}\0{int(environment_seed)}\0"
        f"{int(decision_index)}\0{int(branch)}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % _MAX_TORCH_SEED
