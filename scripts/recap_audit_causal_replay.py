#!/usr/bin/env python3
"""Freeze causal holdout stimuli and audit null replays against the selected states.

Environment seed equality alone is not intervention-state equality. Preserve the
literal collected instruction, continuation noise, and intervention index. A
fresh full-policy evaluation may change these, but must not be called a replay.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def _bool(value):
    if value in (True, "true", "True", "1"):
        return True
    if value in (False, "false", "False", "0"):
        return False
    raise ValueError(f"Missing/invalid common_noise_per_episode: {value!r}")


def build_protocol(inputs: list[Path], *, policy_seed: int) -> dict:
    states = {}
    task_decisions = {}
    schedules = set()
    for root in inputs:
        paths = sorted(root.expanduser().resolve().rglob("manifest.json"))
        if not paths:
            raise FileNotFoundError(f"No paired manifests under {root}")
        for path in paths:
            row = json.loads(path.read_text())
            m = row["metadata"]
            task = m["task_name"]
            seed = int(m["paired_environment_seed"])
            decision = int(m["paired_original_decision_index"])
            if decision < 0 or int(m["counterfactual_policy_decision"]) != decision:
                raise ValueError(f"Inconsistent intervention decision: {path}")
            if not isinstance(task, str) or not task or not isinstance(row["task"], str) or not row["task"] or len(row["steps"]) != 1:
                raise ValueError(f"Expected one selected decision and task text: {path}")
            if not isinstance(row["success"], bool):
                raise ValueError(f"Expected a boolean outcome: {path}")
            if int(row["seed"]) != seed:
                raise ValueError(f"Environment seed mismatch: {path}")
            if task_decisions.setdefault(task, decision) != decision:
                raise ValueError(f"Multiple intervention indices for {task}; split by decision")
            schedule = (int(m["continuation_policy_seed"]), _bool(m["common_noise_per_episode"]))
            schedules.add(schedule)
            key = (task, seed)
            entry = states.setdefault(key, {"instruction": row["task"], "sources": [], "outcomes": set()})
            if entry["instruction"] != row["task"]:
                raise ValueError(f"Conflicting collected instructions: {key}")
            entry["sources"].append(str(path))
            entry["outcomes"].add(row["success"])
    if len(schedules) != 1:
        raise ValueError("Different/missing continuation schedules; split replay runs by schedule")
    continuation_seed, per_episode = schedules.pop()
    if not per_episode:
        raise ValueError("Causal holdout replay requires per-episode common noise")
    tasks = {}
    for (task, seed), entry in sorted(states.items()):
        if entry.pop("outcomes") != {True, False}:
            raise ValueError(f"Not a mixed-outcome state: {(task, seed)}")
        tasks.setdefault(task, {})[str(seed)] = entry
    return {
        "schema_version": 1,
        "purpose": "causal_holdout_replay_not_fresh_policy_validation",
        "policy_seed": policy_seed,
        "continuation_policy_seed": continuation_seed,
        "common_noise_per_episode": per_episode,
        "counterfactual_decisions": task_decisions,
        "tasks": tasks,
    }


def protocol_maps(protocol: dict) -> dict:
    tasks = protocol["tasks"]
    return {
        "environment_seeds.json": {"schema_version": 1, "tasks": {
            task: sorted(map(int, states)) for task, states in tasks.items()
        }},
        "instructions.json": {"schema_version": 1, "tasks": {
            task: {seed: entry["instruction"] for seed, entry in states.items()}
            for task, states in tasks.items()
        }},
        "decisions.json": {"schema_version": 1, "tasks": protocol["counterfactual_decisions"]},
    }


def _attempt(path):
    return max((int(m[1]) for part in path.parts if (m := re.fullmatch(r"attempt-(\d+)", part))), default=0)


def audit_replay(protocol: dict, rollout_root: Path, *, observation_atol=0.001, image_mae_tolerance=2.0) -> dict:
    if observation_atol < 0 or image_mae_tolerance < 0:
        raise ValueError("Tolerances must be non-negative")
    actual = {}
    for path in sorted(rollout_root.rglob("manifest.json")):
        row = json.loads(path.read_text())
        key = (row["metadata"]["task_name"], str(row["seed"]))
        previous = actual.get(key)
        if previous and _attempt(previous[0]) == _attempt(path):
            raise ValueError(f"Duplicate task/seed in the same attempt: {key}")
        if previous is None or _attempt(path) > _attempt(previous[0]):
            actual[key] = (path, row)
    expected = {(task, seed) for task, states in protocol["tasks"].items() for seed in states}
    results = []
    for task, seed in sorted(expected):
        entry = protocol["tasks"][task][seed]
        report = {"task": task, "seed": int(seed), "mismatches": []}
        errors = report["mismatches"]
        results.append(report)
        if (task, seed) not in actual:
            errors.append("missing_episode")
            continue
        path, row = actual[(task, seed)]
        m = row["metadata"]
        decision = protocol["counterfactual_decisions"][task]
        if m.get("recap_condition") != "null":
            raise ValueError("Audit the null replay; an active adapter can intentionally change states")
        if row["task"] != entry["instruction"]:
            errors.append("instruction")
        for key, expected_value in (
            ("policy_seed", protocol["policy_seed"]),
            ("continuation_policy_seed", protocol["continuation_policy_seed"]),
            ("counterfactual_policy_decision", decision),
        ):
            if str(m.get(key)) != str(expected_value):
                errors.append(key)
        if _bool(m.get("common_noise_per_episode")) != protocol["common_noise_per_episode"]:
            errors.append("common_noise_per_episode")
        matches = [s for s in row["steps"] if int(s["decision_index"]) == decision]
        if len(matches) != 1:
            errors.append("intervention_not_reached")
            continue
        distances = []
        with np.load(path.parent / matches[0]["file"], allow_pickle=False) as candidate:
            for source in entry["sources"]:
                source = Path(source)
                original = json.loads(source.read_text())
                with np.load(source.parent / original["steps"][0]["file"], allow_pickle=False) as reference:
                    keys = {k for k in reference.files if k.startswith("observation::")}
                    if not keys or keys != {k for k in candidate.files if k.startswith("observation::")}:
                        raise ValueError("Observation keys missing or different")
                    numeric, image = 0.0, 0.0
                    for key in keys:
                        a, b = reference[key], candidate[key]
                        if a.shape != b.shape or a.dtype != b.dtype:
                            raise ValueError(f"Observation shape/dtype mismatch: {key}")
                        delta = np.abs(a.astype(float) - b.astype(float))
                        if not np.isfinite(delta).all():
                            raise ValueError("Non-finite observation")
                        if "images" in key:
                            image = max(image, float(delta.mean()))
                        else:
                            numeric = max(numeric, float(delta.max()))
                    distances.append((numeric, image))
        report["nearest_numeric_max_abs"] = min(d[0] for d in distances)
        report["nearest_image_mae"] = min(d[1] for d in distances)
        # Both tolerances must hold against the SAME collected observation.
        if not any(n <= observation_atol and i <= image_mae_tolerance for n, i in distances):
            errors.append("intervention_observation")
    counts = defaultdict(int)
    for row in results:
        for error in row["mismatches"]:
            counts[error] += 1
    unexpected = sorted(actual.keys() - expected)
    return {
        "schema_version": 1,
        "passed": not counts and not unexpected,
        "episodes": len(expected),
        "matched_states": sum(not row["mismatches"] for row in results),
        "mismatch_counts": dict(counts),
        "unexpected_task_seeds": unexpected,
        "observation_atol": observation_atol,
        "image_mae_tolerance": image_mae_tolerance,
        "states": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path, help="Paired holdout split")
    parser.add_argument("--policy-seed", required=True, type=int, help="Only changes intervention noise")
    parser.add_argument("--output", required=True, type=Path, help="New protocol directory")
    parser.add_argument("--null-rollouts", type=Path, help="Optionally audit an existing null replay")
    parser.add_argument("--observation-atol", type=float, default=0.001)
    parser.add_argument("--image-mae-tolerance", type=float, default=2.0)
    args = parser.parse_args()
    protocol = build_protocol(args.input, policy_seed=args.policy_seed)
    outputs = {"protocol.json": protocol, **protocol_maps(protocol)}
    if args.null_rollouts:
        outputs["audit.json"] = audit_replay(
            protocol, args.null_rollouts,
            observation_atol=args.observation_atol, image_mae_tolerance=args.image_mae_tolerance,
        )
    args.output.mkdir(parents=True, exist_ok=False)
    for name, payload in outputs.items():
        (args.output / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(outputs.get("audit.json", {"output": str(args.output)}), sort_keys=True))
    if args.null_rollouts and not outputs["audit.json"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
