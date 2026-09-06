#!/usr/bin/env python3
"""Analyze positive/null/signed-negative RoboTwin validation rollouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CONDITIONS = ("positive", "null", "negative")


def _sign_test(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = min(improved, regressed)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / 2**discordant
    return min(1.0, 2.0 * probability)


def analyze(
    input_root: Path,
    seed_indices: list[int],
    *,
    expected_per_task: int,
) -> dict:
    input_root = input_root.expanduser().resolve()
    rows: dict[str, dict[tuple[str, int], dict]] = {condition: {} for condition in CONDITIONS}
    for condition in CONDITIONS:
        for seed_index in seed_indices:
            root = input_root / f"{condition}_s{seed_index}"
            manifests = sorted(root.rglob("manifest.json"))
            if not manifests:
                raise FileNotFoundError(f"No manifests under {root}")
            for path in manifests:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                metadata = manifest.get("metadata") or {}
                task = metadata.get("task_name")
                if not isinstance(task, str) or not task:
                    raise ValueError(f"Missing canonical task name: {path}")
                recorded_condition = metadata.get("recap_condition")
                if recorded_condition != condition:
                    raise ValueError(
                        f"Condition mismatch in {path}: {recorded_condition!r} != {condition!r}"
                    )
                key = (task, int(manifest["seed"]))
                if key in rows[condition]:
                    raise ValueError(f"Duplicate {condition} task/seed: {key}")
                rows[condition][key] = {
                    "success": bool(manifest["success"]),
                    "instruction": str(manifest["task"]),
                    "manifest": str(path),
                }

    task_sets = [set(task for task, _ in condition_rows) for condition_rows in rows.values()]
    if any(tasks != task_sets[0] for tasks in task_sets[1:]):
        raise ValueError("Task sets differ across validation conditions")

    result = {
        "schema_version": 1,
        "input_root": str(input_root),
        "conditions": list(CONDITIONS),
        "seed_indices": seed_indices,
        "expected_per_task": expected_per_task,
        "tasks": {},
    }
    for task in sorted(task_sets[0]):
        per_condition = {
            condition: {
                seed: row
                for (row_task, seed), row in rows[condition].items()
                if row_task == task
            }
            for condition in CONDITIONS
        }
        raw = {
            condition: {
                "success": sum(row["success"] for row in condition_rows.values()),
                "episodes": len(condition_rows),
            }
            for condition, condition_rows in per_condition.items()
        }
        common = set.intersection(*(set(values) for values in per_condition.values()))
        instruction_mismatches = []
        for seed in sorted(common):
            instructions = {
                condition: per_condition[condition][seed]["instruction"]
                for condition in CONDITIONS
            }
            if len(set(instructions.values())) != 1:
                instruction_mismatches.append({"seed": seed, "instructions": instructions})
        if instruction_mismatches:
            raise ValueError(
                f"Instruction mismatch for {task}: {instruction_mismatches[:3]}"
            )
        paired_success = {
            condition: sum(per_condition[condition][seed]["success"] for seed in common)
            for condition in CONDITIONS
        }

        def comparison(left: str, right: str) -> dict:
            improved = sum(
                per_condition[left][seed]["success"]
                and not per_condition[right][seed]["success"]
                for seed in common
            )
            regressed = sum(
                per_condition[right][seed]["success"]
                and not per_condition[left][seed]["success"]
                for seed in common
            )
            return {
                "improved": improved,
                "regressed": regressed,
                "ties": len(common) - improved - regressed,
                "two_sided_sign_test_p": _sign_test(improved, regressed),
            }

        complete = (
            len(common) == expected_per_task
            and all(value["episodes"] == expected_per_task for value in raw.values())
        )
        strict_order = (
            paired_success["positive"]
            > paired_success["null"]
            > paired_success["negative"]
        )
        promoted = complete and strict_order
        reasons = []
        if not complete:
            reasons.append(
                f"only {len(common)}/{expected_per_task} seeds are common across conditions"
            )
        if not strict_order:
            reasons.append("paired success does not satisfy positive > null > negative")
        result["tasks"][task] = {
            "raw": raw,
            "paired_common_episodes": len(common),
            "paired_success": paired_success,
            "positive_vs_null": comparison("positive", "null"),
            "null_vs_negative": comparison("null", "negative"),
            "strict_order": strict_order,
            "promoted": promoted,
            "rejection_reasons": reasons,
        }

    result["promoted_tasks"] = sorted(
        task for task, value in result["tasks"].items() if value["promoted"]
    )
    result["rejected_tasks"] = sorted(
        task for task, value in result["tasks"].items() if not value["promoted"]
    )
    result["paired_totals"] = {
        condition: {
            "success": sum(
                value["paired_success"][condition] for value in result["tasks"].values()
            ),
            "episodes": sum(
                value["paired_common_episodes"] for value in result["tasks"].values()
            ),
        }
        for condition in CONDITIONS
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--seed-index", action="append", required=True, type=int)
    parser.add_argument("--expected-per-task", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(
        args.input_root,
        args.seed_index,
        expected_per_task=args.expected_per_task,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "promoted_tasks": result["promoted_tasks"],
                "rejected_tasks": result["rejected_tasks"],
                "paired_totals": result["paired_totals"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
