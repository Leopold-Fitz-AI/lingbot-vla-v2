"""Batch-size-independent RoboTwin instruction generation (no simulator imports)."""

from __future__ import annotations

import hashlib
import random
from copy import deepcopy

import numpy as np


# This is part of the evaluation protocol, NOT the number of rollout episodes.
# Changing it changes task text; replay old datasets using their instruction map.
DETERMINISTIC_CANDIDATES = 32
INSTRUCTION_PROTOCOL = "task-seed-v2-fixed-candidates32"


def generate_task_instruction(
    task_name,
    seed,
    episode_info_list,
    instruction_type,
    test_num,
    *,
    deterministic,
    generator,
):
    """Generate one instruction without coupling deterministic text to run size.

    RoboTwin's generator's third argument is the number of *descriptions*.
    Passing test_num there, then hashing modulo the resulting list length, made
    the old deterministic mode change text between collection and small holdouts.
    Legacy random mode deliberately retains its original behavior.
    """
    if not deterministic:
        results = generator(task_name, episode_info_list, test_num)
        return str(np.random.choice(results[0][instruction_type]))

    generation_seed = int.from_bytes(
        hashlib.sha256(f"instruction-generation:{task_name}:{seed}".encode()).digest()[:8],
        "big",
    )
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(generation_seed)
        np.random.seed(generation_seed % 2**32)
        results = generator(
            task_name, deepcopy(episode_info_list), DETERMINISTIC_CANDIDATES
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    if not results or not results[0].get(instruction_type):
        raise ValueError(f"No instruction candidates for task={task_name!r}, seed={seed}")
    candidates = results[0][instruction_type]
    digest = hashlib.sha256(f"{task_name}:{seed}".encode()).digest()
    index = int.from_bytes(digest[:8], "big") % len(candidates)
    return str(candidates[index])
