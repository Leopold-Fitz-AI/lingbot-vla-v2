"""Validation protocol manifests shared by RECAP launchers and clients."""

from __future__ import annotations

import json
from pathlib import Path


def load_environment_seed_map(path: str | Path) -> dict[str, list[int]]:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("tasks"), dict):
        raise ValueError(
            "Environment seed map must have schema_version=1 and a tasks object"
        )
    result = {}
    for task, seeds in payload["tasks"].items():
        if not isinstance(task, str) or not task:
            raise ValueError("Environment seed map task names must be non-empty strings")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"Environment seed map for task={task!r} must be non-empty")
        if any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            for seed in seeds
        ):
            raise ValueError(f"Environment seed map for task={task!r} is invalid")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Environment seed map for task={task!r} has duplicates")
        result[task] = list(seeds)
    return result
