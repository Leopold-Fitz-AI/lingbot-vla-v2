#!/usr/bin/env python3
"""Split paired causal rollouts by independent environment state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path


def _rank(task: str, seed: int, split_seed: int) -> str:
    return hashlib.sha256(f"{split_seed}\0{task}\0{seed}".encode()).hexdigest()


def split_paired_states(
    input_root: Path,
    output_root: Path,
    *,
    task_name: str,
    holdout_fraction: float = 0.25,
    minimum_train_states: int = 8,
    minimum_holdout_states: int = 2,
    split_seed: int = 42,
) -> dict:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between zero and one")
    if minimum_train_states <= 0 or minimum_holdout_states <= 0:
        raise ValueError("minimum state counts must be positive")
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    groups = defaultdict(list)
    for path in sorted(input_root.rglob("manifest.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        metadata = row.get("metadata") or {}
        if metadata.get("task_name") != task_name:
            raise ValueError(f"Unexpected task in {path}")
        seed = int(metadata["paired_environment_seed"])
        pair_id = str(metadata["paired_match_id"])
        groups[(seed, pair_id)].append((path, row))
    if not groups:
        raise FileNotFoundError(f"No paired manifests under {input_root}")
    by_state = defaultdict(list)
    for (seed, pair_id), members in groups.items():
        if len(members) != 2 or {bool(row[1]["success"]) for row in members} != {False, True}:
            raise ValueError(f"Incomplete signed pair for state {seed}: {pair_id}")
        by_state[seed].extend(members)
    states = sorted(by_state, key=lambda seed: (_rank(task_name, seed, split_seed), seed))
    holdout_count = max(minimum_holdout_states, math.ceil(len(states) * holdout_fraction))
    if len(states) - holdout_count < minimum_train_states:
        raise ValueError(
            f"Only {len(states)} states cannot provide {minimum_train_states} train and "
            f"{holdout_count} holdout states"
        )
    holdout_states = set(states[:holdout_count])
    assignments = {
        "train": set(states) - holdout_states,
        "holdout": holdout_states,
    }
    temporary = output_root.with_name(f".{output_root.name}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    try:
        for split, split_states in assignments.items():
            destination = temporary / split
            destination.mkdir(parents=True)
            episodes = []
            index = 0
            for seed in sorted(split_states):
                for source_manifest, row in sorted(
                    by_state[seed], key=lambda value: value[1]["episode_id"]
                ):
                    target = destination / f"episode-{index:06d}"
                    shutil.copytree(source_manifest.parent, target)
                    episodes.append(
                        {
                            "episode_id": row["episode_id"],
                            "task": row["task"],
                            "success": bool(row["success"]),
                            "steps": [
                                {
                                    "decision_index": 0,
                                    "executed_action_length": row["steps"][0]["executed_action_length"],
                                    "terminated": row["steps"][0].get("terminated", False),
                                    "valid": True,
                                }
                            ],
                        }
                    )
                    index += 1
            with (destination / "episodes.jsonl").open("w", encoding="utf-8") as handle:
                for episode in episodes:
                    handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
            (destination / "split.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "split": split,
                        "task_name": task_name,
                        "environment_seeds": sorted(split_states),
                        "independent_states": len(split_states),
                        "episodes": len(episodes),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        summary = {
            "schema_version": 1,
            "task_name": task_name,
            "input": str(input_root),
            "split_seed": split_seed,
            "holdout_fraction": holdout_fraction,
            "train_states": sorted(assignments["train"]),
            "holdout_states": sorted(assignments["holdout"]),
            "train_state_count": len(assignments["train"]),
            "holdout_state_count": len(assignments["holdout"]),
        }
        (temporary / "split_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-train-states", type=int, default=8)
    parser.add_argument("--minimum-holdout-states", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            split_paired_states(
                args.input,
                args.output,
                task_name=args.task,
                holdout_fraction=args.holdout_fraction,
                minimum_train_states=args.minimum_train_states,
                minimum_holdout_states=args.minimum_holdout_states,
                split_seed=args.split_seed,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
