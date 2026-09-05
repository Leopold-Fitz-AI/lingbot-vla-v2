#!/usr/bin/env python3
"""Select balanced causal action pairs for every task in a decision map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.recap_select_paired_decisions import select_paired_decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--decision-map", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--observation-atol", type=float, default=0.0)
    parser.add_argument("--image-mae-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--pairwise-match",
        action="store_true",
        help="retain only tolerance-valid matched success/failure pairs",
    )
    parser.add_argument("--minimum-positive", type=int, default=1)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Output already exists: {args.output_root}")
    if args.minimum_positive <= 0:
        raise ValueError("minimum-positive must be positive")
    with args.decision_map.open() as handle:
        decision_map = json.load(handle)
    if decision_map.get("schema_version") != 1 or not isinstance(
        decision_map.get("tasks"), dict
    ):
        raise ValueError("decision-map must have schema_version=1 and a tasks object")

    args.output_root.mkdir(parents=True)
    selected = {}
    insufficient = {}
    rejected = {}
    for task, decision in sorted(decision_map["tasks"].items()):
        task_output = args.output_root / task
        try:
            summary = select_paired_decisions(
                args.input,
                task_output,
                decision_index=int(decision),
                task_name=task,
                balance_outcomes=True,
                observation_atol=args.observation_atol,
                image_mae_tolerance=args.image_mae_tolerance,
                pairwise_match=args.pairwise_match,
            )
        except ValueError as error:
            message = str(error)
            if "No state group contains both" in message:
                insufficient[task] = message
                continue
            if (
                "exceed pairing tolerance" in message
                or "pair satisfies pairing tolerance" in message
                or "Task mismatch" in message
                or "identical" in message
            ):
                rejected[task] = message
                continue
            raise
        if summary["positive"] < args.minimum_positive:
            insufficient[task] = (
                f"only {summary['positive']} positive samples; "
                f"minimum is {args.minimum_positive}"
            )
        else:
            selected[task] = summary

    result = {
        "schema_version": 1,
        "inputs": [str(path.resolve()) for path in args.input],
        "decision_map": str(args.decision_map.resolve()),
        "selected_tasks": selected,
        "insufficient_tasks": insufficient,
        "rejected_tasks": rejected,
        "selected_task_count": len(selected),
        "insufficient_task_count": len(insufficient),
        "rejected_task_count": len(rejected),
    }
    (args.output_root / "selection_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
