#!/usr/bin/env python3
"""Plan task-specific counterfactual decision interventions from reference rollouts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def _attempt_number(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"attempt-(\d+)", part)
        if match:
            return int(match.group(1))
    return 0


def _load_reference_manifests(root: Path) -> dict[str, list[tuple[Path, dict]]]:
    latest: dict[tuple[str, int], tuple[int, Path, dict]] = {}
    for path in root.rglob("manifest.json"):
        with path.open() as handle:
            manifest = json.load(handle)
        metadata = manifest.get("metadata") or {}
        task = metadata.get("task_name")
        seed = manifest.get("seed")
        if not isinstance(task, str) or not isinstance(seed, int):
            raise ValueError(f"Manifest lacks task_name or integer seed: {path}")
        candidate = (_attempt_number(path), path, manifest)
        key = (task, seed)
        if key not in latest or candidate[0] > latest[key][0]:
            latest[key] = candidate

    grouped: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for (task, _), (_, path, manifest) in latest.items():
        grouped[task].append((path, manifest))
    for rows in grouped.values():
        rows.sort(key=lambda item: item[1]["seed"])
    return dict(grouped)


def _load_baseline_rates(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    if path.is_file():
        paths = [path]
    else:
        paths = sorted(path.glob("*/_result.json"))
    result = {}
    for result_path in paths:
        with result_path.open() as handle:
            row = json.load(handle)
        result[row["task"]] = {
            "success": int(row["success"]),
            "episodes": int(row["episodes"]),
            "success_rate": float(row["success_rate"]),
        }
    return result


def _recommended_decision(manifests: list[dict]) -> tuple[int, str, dict]:
    successful_terminal = [
        int(manifest["steps"][-1]["decision_index"])
        for manifest in manifests
        if manifest.get("success") and manifest.get("steps")
    ]
    if successful_terminal:
        counts = Counter(successful_terminal)
        maximum = max(counts.values())
        modes = sorted(index for index, count in counts.items() if count == maximum)
        choice = min(modes, key=lambda index: (abs(index - median(successful_terminal)), index))
        return choice, "successful_terminal_mode", dict(sorted(counts.items()))

    decision_counts = [
        len(manifest.get("steps") or [])
        for manifest in manifests
        if manifest.get("steps")
    ]
    if not decision_counts:
        raise ValueError("Cannot plan a decision from episodes with no recorded steps")
    # With no successful reference, avoid the final repeated step-limit action.
    choice = max(0, min(2, round(median(decision_counts)) - 2))
    return choice, "failure_horizon_fallback", {}


def plan_task_interventions(
    rollout_root: Path,
    *,
    baseline_results: Path | None = None,
    minimum_episodes: int = 1,
) -> tuple[dict, dict, dict]:
    grouped = _load_reference_manifests(rollout_root)
    baseline = _load_baseline_rates(baseline_results)
    tasks = {}
    decision_tasks = {}
    instruction_tasks = {}

    for task, rows in sorted(grouped.items()):
        manifests = [manifest for _, manifest in rows]
        if len(manifests) < minimum_episodes:
            raise ValueError(
                f"Task {task!r} has {len(manifests)} episodes, expected at least {minimum_episodes}"
            )
        decision, source, terminal_histogram = _recommended_decision(manifests)
        success = sum(bool(manifest.get("success")) for manifest in manifests)
        episode_count = len(manifests)
        baseline_row = baseline.get(task)
        guard = bool(
            baseline_row is not None and baseline_row["success_rate"] >= 1.0
        )
        candidates = sorted({max(0, decision - 1), decision, decision + 1})
        tasks[task] = {
            "reference_success": success,
            "reference_episodes": episode_count,
            "reference_success_rate": success / episode_count,
            "baseline": baseline_row,
            "perfect_baseline_guard": guard,
            "recommended_decision": decision,
            "decision_source": source,
            "candidate_decisions": candidates,
            "successful_terminal_histogram": {
                str(key): value for key, value in terminal_histogram.items()
            },
            "decision_count_histogram": {
                str(key): value
                for key, value in sorted(
                    Counter(len(manifest.get("steps") or []) for manifest in manifests).items()
                )
            },
        }
        if not guard:
            decision_tasks[task] = decision
            instruction_tasks[task] = {
                str(manifest["seed"]): manifest["task"]
                for manifest in manifests
            }

    plan = {
        "schema_version": 1,
        "rollout_root": str(rollout_root.resolve()),
        "minimum_episodes": minimum_episodes,
        "tasks": tasks,
        "target_tasks": sorted(decision_tasks),
        "guard_tasks": sorted(task for task, row in tasks.items() if row["perfect_baseline_guard"]),
    }
    decision_map = {"schema_version": 1, "tasks": decision_tasks}
    instruction_map = {"schema_version": 1, "tasks": instruction_tasks}
    return plan, decision_map, instruction_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-episodes", type=int, default=1)
    args = parser.parse_args()
    if args.minimum_episodes <= 0:
        raise ValueError("minimum-episodes must be positive")
    plan, decision_map, instruction_map = plan_task_interventions(
        args.rollout_root,
        baseline_results=args.baseline_results,
        minimum_episodes=args.minimum_episodes,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "plan.json": plan,
        "counterfactual_decisions.json": decision_map,
        "instructions.json": instruction_map,
    }
    for name, payload in outputs.items():
        (args.output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    (args.output_dir / "target_tasks.txt").write_text(
        "\n".join(plan["target_tasks"]) + "\n"
    )
    (args.output_dir / "target_tasks.csv").write_text(
        ",".join(plan["target_tasks"]) + "\n"
    )
    print(
        json.dumps(
            {
                "tasks": len(plan["tasks"]),
                "target_tasks": len(plan["target_tasks"]),
                "guard_tasks": len(plan["guard_tasks"]),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
