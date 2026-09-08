#!/usr/bin/env python3
"""Compare named policy variants on a frozen set of paired rollout manifests.

Descriptive diagnostic only: this does not promote adapters or turn an
underpowered, non-significant comparison into evidence of no effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def _attempt(path):
    return max((int(m[1]) for part in path.parts if (m := re.fullmatch(r"attempt-(\d+)", part))), default=0)


def compare_variants(variants: dict[str, Path], *, reference: str, seed_map: dict) -> dict:
    if reference not in variants or len(variants) < 2:
        raise ValueError("At least two variants including the reference are required")
    if seed_map.get("schema_version") != 1 or not seed_map.get("tasks"):
        raise ValueError("Expected schema_version=1 and a nonempty tasks seed map")
    expected = {(task, int(seed)) for task, seeds in seed_map["tasks"].items() for seed in seeds}
    if len(expected) != sum(len(seeds) for seeds in seed_map["tasks"].values()):
        raise ValueError("Duplicate planned seeds")
    rows = {}
    for name, root in variants.items():
        selected = {}
        for path in sorted(root.rglob("manifest.json")):
            encoded = path.read_bytes()
            row = json.loads(encoded)
            key = (row["metadata"]["task_name"], int(row["seed"]))
            previous = selected.get(key)
            if previous and _attempt(previous["path"]) == _attempt(path):
                raise ValueError(f"Duplicate task/seed for {name}: {key}")
            if previous is None or _attempt(path) > _attempt(previous["path"]):
                if not isinstance(row["success"], bool):
                    raise ValueError(f"Non-boolean outcome: {path}")
                selected[key] = {"path": path, "manifest": row, "sha256": hashlib.sha256(encoded).hexdigest()}
        if set(selected) != expected:
            raise ValueError(f"{name}: seed set mismatch; missing={sorted(expected - selected.keys())}, "
                             f"unexpected={sorted(selected.keys() - expected)}")
        rows[name] = selected
    for key in sorted(expected):
        base = rows[reference][key]["manifest"]
        for name in variants:
            row = rows[name][key]["manifest"]
            if row["task"] != base["task"]:
                raise ValueError(f"Instruction mismatch for {name}/{key}")
            for field in ("policy_seed", "continuation_policy_seed", "counterfactual_policy_decision",
                          "common_noise_per_episode", "task_config", "step_limit"):
                value = row["metadata"].get(field)
                if value is None or value == "" or str(value) != str(base["metadata"].get(field)):
                    raise ValueError(f"Protocol mismatch/missing {field} for {name}/{key}")
            if str(row["metadata"]["common_noise_per_episode"]).lower() not in ("true", "1"):
                raise ValueError("Paired comparison requires per-episode common action noise")
    tasks = {}
    for task in sorted(seed_map["tasks"]):
        keys = sorted(key for key in expected if key[0] == task)
        outcomes = {name: [rows[name][key]["manifest"]["success"] for key in keys] for name in variants}
        result = {"episodes": len(keys), "success": {name: sum(values) for name, values in outcomes.items()},
                  "comparisons_vs_reference": {}}
        for name, values in outcomes.items():
            if name == reference:
                continue
            improved = sum(a and not b for a, b in zip(values, outcomes[reference]))
            regressed = sum(b and not a for a, b in zip(values, outcomes[reference]))
            n = improved + regressed
            p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(improved, regressed) + 1)) / 2**n)
            result["comparisons_vs_reference"][name] = {
                "improved": improved, "regressed": regressed, "ties": len(keys) - n,
                "two_sided_sign_test_p": p,
                "evidence": "inconclusive" if p >= 0.05 else ("uplift" if improved > regressed else "regression"),
            }
        tasks[task] = result
    return {"schema_version": 1, "purpose": "diagnostic_not_promotion", "reference": reference, "tasks": tasks,
            "provenance": {name: [{"task": key[0], "seed": key[1], "manifest": str(value["path"]),
                                    "sha256": value["sha256"]} for key, value in sorted(values.items())]
                           for name, values in rows.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True, help="NAME=ROLLOUT_DIRECTORY")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--environment-seed-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    variants = {}
    for value in args.variant:
        name, root = value.split("=", 1)
        if not name or name in variants:
            raise ValueError(f"Invalid/duplicate variant name: {name!r}")
        variants[name] = Path(root).expanduser().resolve()
    result = compare_variants(variants, reference=args.reference,
                              seed_map=json.loads(args.environment_seed_map.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["tasks"], sort_keys=True))


if __name__ == "__main__":
    main()
