#!/usr/bin/env python3
"""Select outcome-diverse actions sampled from identical rollout states.

RECAP needs action outcomes that vary while state/task stay fixed. Ordinary
success/failure trajectory relabeling confounds action quality with environment
and state difficulty. This utility groups repeated rollouts by environment seed,
keeps only groups containing both outcomes, verifies that the selected decision
observations are byte-identical, and emits one decision per source episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def _manifest_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("manifest.json"))


def _attempt_number(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"attempt-(\d+)", part)
        if match:
            return int(match.group(1))
    return 0


def _step(manifest: dict, decision_index: int) -> dict:
    matches = [
        step
        for step in manifest.get("steps", [])
        if int(step.get("decision_index", -1)) == decision_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one decision {decision_index}, found {len(matches)} "
            f"in episode {manifest.get('episode_id')!r}"
        )
    return matches[0]


def _array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _load_arrays(path: Path) -> tuple[dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as arrays:
        observation = {
            key: np.asarray(arrays[key]).copy()
            for key in arrays.files
            if key.startswith("observation::")
        }
        if "executed_action" not in arrays.files:
            raise ValueError(f"{path} has no executed_action")
        action = _array_digest(arrays["executed_action"])
    if not observation:
        raise ValueError(f"{path} has no observation arrays")
    return observation, action


def _observations_match(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    numeric_atol: float,
    image_mae_tolerance: float,
) -> bool:
    if reference.keys() != candidate.keys():
        return False
    for key, expected in reference.items():
        actual = candidate[key]
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            return False
        if np.array_equal(expected, actual):
            continue
        difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
        if "images" in key:
            if float(difference.mean()) > image_mae_tolerance:
                return False
        elif float(difference.max()) > numeric_atol:
            return False
    return True


def select_paired_decisions(
    inputs: list[Path],
    output: Path,
    *,
    decision_index: int = 0,
    balance_outcomes: bool = False,
    observation_atol: float = 0.0,
    image_mae_tolerance: float = 0.0,
    task_name: str | None = None,
) -> dict:
    if len(inputs) < 2:
        raise ValueError("At least two independently sampled rollout roots are required")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    groups: dict[int, list[dict]] = defaultdict(list)
    seen_sources: set[str] = set()
    for root in inputs:
        root = root.expanduser().resolve()
        source_name = root.name
        if source_name in seen_sources:
            raise ValueError(f"Duplicate rollout source name: {source_name}")
        seen_sources.add(source_name)
        paths = _manifest_paths(root)
        if not paths:
            raise ValueError(f"No manifests found under {root}")
        matched_paths = 0
        root_rows = {}
        for path in paths:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest_task_name = (manifest.get("metadata") or {}).get("task_name")
            if task_name is not None and manifest_task_name != task_name:
                continue
            matched_paths += 1
            seed = manifest.get("seed")
            if seed is None:
                raise ValueError(f"Manifest has no environment seed: {path}")
            if not any(
                int(item.get("decision_index", -1)) == decision_index
                for item in manifest.get("steps", [])
            ):
                continue
            step = _step(manifest, decision_index)
            row = {
                "root": root,
                "source": source_name,
                "manifest_path": path,
                "manifest": manifest,
                "step": step,
                "npz": path.parent / step["file"],
            }
            key = int(seed)
            previous = root_rows.get(key)
            if previous is None or _attempt_number(path) > _attempt_number(
                previous["manifest_path"]
            ):
                root_rows[key] = row
        if task_name is not None and matched_paths == 0:
            raise ValueError(f"No manifests for task {task_name!r} under {root}")
        for seed, row in root_rows.items():
            groups[seed].append(row)

    selected_groups = {
        seed: rows
        for seed, rows in groups.items()
        if {bool(row["manifest"].get("success")) for row in rows} == {False, True}
    }
    if not selected_groups:
        raise ValueError("No state group contains both successful and failed actions")
    if balance_outcomes:
        balanced_groups = {}
        for seed, rows in selected_groups.items():
            successes = sorted(
                (row for row in rows if row["manifest"].get("success")),
                key=lambda row: row["source"],
            )
            failures = sorted(
                (row for row in rows if not row["manifest"].get("success")),
                key=lambda row: row["source"],
            )
            count = min(len(successes), len(failures))
            balanced_groups[seed] = successes[:count] + failures[:count]
        selected_groups = balanced_groups

    temporary = output.with_name(f".{output.name}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    episodes = []
    group_summaries = []
    positive = negative = 0
    try:
        output_index = 0
        for seed, rows in sorted(selected_groups.items()):
            tasks = {str(row["manifest"].get("task")) for row in rows}
            if len(tasks) != 1:
                raise ValueError(f"Task mismatch for seed {seed}: {sorted(tasks)}")
            reference_observation = None
            action_digests: set[str] = set()
            outcomes = []
            for row in rows:
                observation, action_digest = _load_arrays(row["npz"])
                if reference_observation is None:
                    reference_observation = observation
                elif not _observations_match(
                    reference_observation,
                    observation,
                    numeric_atol=observation_atol,
                    image_mae_tolerance=image_mae_tolerance,
                ):
                    raise ValueError(
                        f"Selected observations exceed pairing tolerance for seed {seed}"
                    )
                action_digests.add(action_digest)
                outcomes.append(bool(row["manifest"].get("success")))
            if len(action_digests) < 2:
                raise ValueError(f"All sampled actions are identical for seed {seed}")

            group_summaries.append(
                {
                    "seed": seed,
                    "samples": len(rows),
                    "successes": sum(outcomes),
                    "failures": len(outcomes) - sum(outcomes),
                    "unique_actions": len(action_digests),
                }
            )
            for row in rows:
                original = row["manifest"]
                success = bool(original.get("success"))
                positive += int(success)
                negative += int(not success)
                episode_dir = temporary / f"episode-{output_index:06d}"
                steps_dir = episode_dir / "steps"
                steps_dir.mkdir(parents=True)
                relative = "steps/000000.npz"
                shutil.copy2(row["npz"], episode_dir / relative)

                copied_step = dict(row["step"])
                copied_step["decision_index"] = 0
                copied_step["file"] = relative
                copied_step["paired_original_decision_index"] = decision_index
                episode_id = (
                    f"paired-seed-{seed}-{row['source']}-"
                    f"{original.get('episode_id', output_index)}"
                )
                metadata = dict(original.get("metadata") or {})
                metadata.update(
                    {
                        "source": "paired_policy_sample",
                        "paired_environment_seed": seed,
                        "paired_rollout_source": row["source"],
                        "paired_original_manifest": str(row["manifest_path"]),
                        "paired_original_decision_index": decision_index,
                    }
                )
                manifest = {
                    **original,
                    "episode_id": episode_id,
                    "num_decisions": 1,
                    "metadata": metadata,
                    "steps": [copied_step],
                }
                (episode_dir / "manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "task": manifest["task"],
                        "success": success,
                        "steps": [
                            {
                                "decision_index": 0,
                                "executed_action_length": copied_step[
                                    "executed_action_length"
                                ],
                                "terminated": copied_step.get("terminated", False),
                                "valid": True,
                            }
                        ],
                    }
                )
                output_index += 1

        with (temporary / "episodes.jsonl").open("w", encoding="utf-8") as handle:
            for episode in episodes:
                handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
        summary = {
            "schema_version": 1,
            "inputs": [str(path.expanduser().resolve()) for path in inputs],
            "decision_index": decision_index,
            "task_name": task_name,
            "balance_outcomes": balance_outcomes,
            "observation_atol": observation_atol,
            "image_mae_tolerance": image_mae_tolerance,
            "all_environment_seeds": len(groups),
            "mixed_outcome_seeds": len(selected_groups),
            "episodes": len(episodes),
            "positive": positive,
            "negative": negative,
            "groups": group_summaries,
        }
        (temporary / "selection.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**summary, "output": str(output)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decision-index", type=int, default=0)
    parser.add_argument("--task")
    parser.add_argument("--balance-outcomes", action="store_true")
    parser.add_argument("--observation-atol", type=float, default=0.0)
    parser.add_argument("--image-mae-tolerance", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.decision_index < 0:
        raise ValueError("decision-index must be non-negative")
    if args.observation_atol < 0 or args.image_mae_tolerance < 0:
        raise ValueError("pairing tolerances must be non-negative")
    summary = select_paired_decisions(
        [Path(value) for value in args.input],
        Path(args.output).expanduser().resolve(),
        decision_index=args.decision_index,
        balance_outcomes=args.balance_outcomes,
        observation_atol=args.observation_atol,
        image_mae_tolerance=args.image_mae_tolerance,
        task_name=args.task,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
