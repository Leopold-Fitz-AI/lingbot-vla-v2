#!/usr/bin/env python3
"""Freeze exact per-task environment seeds from one or more rollout roots."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path


def build_seed_map(inputs: list[Path], *, seeds_per_task: int) -> dict:
    if seeds_per_task <= 0:
        raise ValueError("seeds_per_task must be positive")
    source_maps = []
    for root in inputs:
        root = root.expanduser().resolve()
        task_seeds = defaultdict(set)
        manifests = sorted(root.rglob("manifest.json"))
        if not manifests:
            raise FileNotFoundError(f"No manifests under {root}")
        for path in manifests:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            task = (manifest.get("metadata") or {}).get("task_name")
            if not isinstance(task, str) or not task:
                raise ValueError(f"Missing canonical task name: {path}")
            task_seeds[task].add(int(manifest["seed"]))
        source_maps.append(task_seeds)
    task_sets = [set(value) for value in source_maps]
    if any(tasks != task_sets[0] for tasks in task_sets[1:]):
        raise ValueError("Input rollout roots contain different task sets")
    selected = {}
    for task in sorted(task_sets[0]):
        common = set.intersection(*(mapping[task] for mapping in source_maps))
        if len(common) < seeds_per_task:
            raise ValueError(
                f"Task {task!r} has only {len(common)} common environment seeds; "
                f"requires {seeds_per_task}"
            )
        selected[task] = sorted(common)[:seeds_per_task]
    return {
        "schema_version": 1,
        "inputs": [str(path.expanduser().resolve()) for path in inputs],
        "seeds_per_task": seeds_per_task,
        "tasks": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--seeds-per-task", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_seed_map(args.input, seeds_per_task=args.seeds_per_task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, args.output)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    print(json.dumps({"tasks": len(result["tasks"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
