"""Predeclared paired inference for a FIXED task suite (not an unseen-task claim)."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from lingbotvla.recap.noise import policy_action_seed


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload, *, immutable=False):
    """Verified local write. Immutable study decisions cannot be overwritten."""
    path = Path(path)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"Frozen document changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if path.read_bytes() != encoded:
        raise IOError(f"JSON read-back mismatch: {path}")


def exact_p(improved, regressed):
    n = improved + regressed
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(improved, regressed) + 1)) / 2**n)


def paired_counts(positive, null):
    if len(positive) != len(null) or not len(null):
        raise ValueError("Equal nonempty paired samples are required")
    a, b = np.asarray(positive, dtype=bool), np.asarray(null, dtype=bool)
    improved, regressed = int((a & ~b).sum()), int((b & ~a).sum())
    return {"episodes": len(a), "positive_success": int(a.sum()), "null_success": int(b.sum()),
            "improved": improved, "regressed": regressed, "ties": len(a) - improved - regressed,
            "delta": (improved - regressed) / len(a), "two_sided_p": exact_p(improved, regressed)}


def holm_adjust(p_values):
    """Family includes ALL predeclared hypotheses, not only promising tasks."""
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted, running = {}, 0.0
    for index, key in enumerate(ordered):
        p = p_values[key]
        if not 0 <= p <= 1:
            raise ValueError("Invalid p-value")
        running = max(running, min(1.0, (len(ordered) - index) * p))
        adjusted[key] = running
    return adjusted


def validate_runtime(row, *, task, seed, instruction, condition, registry, protocol):
    if row["task"] != instruction or row["seed"] != seed or row["metadata"]["task_name"] != task:
        raise ValueError("Task/seed/instruction mismatch")
    if not isinstance(row["success"], bool) or not row.get("steps"):
        raise ValueError("Invalid/empty rollout outcome")
    entry = registry.get("tasks", {}).get(task)
    for index, step in enumerate(row["steps"]):
        runtime = (step.get("info") or {}).get("recap_runtime")
        if not isinstance(runtime, dict) or runtime.get("schema_version") != 1:
            raise ValueError("Missing actual server-reported runtime protocol")
        offset = index - (entry.get("condition_start_decision", 0) if entry else 0)
        count = entry.get("condition_decisions", -1) if entry else -1
        active = entry is not None and offset >= 0 and (count == -1 or offset < count)
        expected = condition if active else "null"
        action_seed = policy_action_seed(
            policy_seed=protocol["policy_seed"],
            continuation_policy_seed=protocol["continuation_policy_seed"],
            counterfactual_decision=protocol["counterfactual_policy_decision"],
            decision_index=index, task_name=task, environment_seed=seed, per_episode=True,
        )
        required = {"task_name": task, "environment_seed": seed, "decision_index": index,
                    "effective_condition": expected, "action_noise_seed": action_seed,
                    "adapter_sha256": entry["sha256"] if entry else None,
                    "condition_start_decision": entry.get("condition_start_decision", 0) if entry else 0,
                    "condition_decisions": entry.get("condition_decisions", -1) if entry else -1,
                    "use_compile": False, "use_bf16": False, "cfg_scale": 1.0}
        if protocol.get("deterministic_algorithms"):
            required.update(deterministic_algorithms=True, deterministic_warn_only=False)
        if not required.keys() <= runtime.keys() or any(runtime[k] != v for k, v in required.items()):
            raise ValueError(f"Actual policy runtime mismatch at decision {index}: {runtime} vs {required}")
        if (step["decision_index"] != index or step["generated_action_length"] != 50
                or not 0 < step["executed_action_length"] <= step["generated_action_length"]):
            raise ValueError("Invalid decision indices/action lengths")
    if bool(row["steps"][-1].get("terminated")) != row["success"]:
        raise ValueError("Terminal flag disagrees with outcome")


def initial_drift(left, right):
    """Arguments are (manifest path, decoded manifest). Only compare BEFORE action."""
    paths = [path.parent / row["steps"][0]["file"] for path, row in (left, right)]
    numeric, image = 0.0, 0.0
    with np.load(paths[0], allow_pickle=False) as a, np.load(paths[1], allow_pickle=False) as b:
        keys = {k for k in a.files if k.startswith("observation::")}
        if not keys or keys != {k for k in b.files if k.startswith("observation::")}:
            raise ValueError("Missing/different observation keys")
        for key in keys:
            x, y = a[key], b[key]
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError("Initial observation shape/dtype mismatch")
            delta = np.abs(x.astype(float) - y.astype(float))
            if not np.isfinite(delta).all():
                raise ValueError("Nonfinite initial observation")
            if "images" in key:
                image = max(image, float(delta.mean()))
            else:
                numeric = max(numeric, float(delta.max()))
    return {"numeric_max_abs": numeric, "image_mae_max": image}


def final_statistics(pairs, *, tasks, registry_tasks, expected_per_task=60,
                     bootstrap_replicates=20000, bootstrap_seed=20260908):
    """pairs: task -> rows with seed_index, positive, null; already protocol-audited.

    Bootstrap samples PAIRS inside each task × seed-index block. Equal task
    sample counts are mandatory for the accompanying pooled exact paired test.
    """
    if len(set(tasks)) != len(tasks) or set(pairs) != set(tasks):
        raise ValueError("The complete frozen task list is required")
    if not set(registry_tasks) <= set(tasks):
        raise ValueError("Unknown registry task")
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.zeros(bootstrap_replicates)
    per_task, total_improved, total_regressed = {}, 0, 0
    for task in tasks:
        rows = pairs[task]
        if len(rows) != expected_per_task:
            raise ValueError(f"Incomplete/unequal cohort for {task}")
        positive, null = [r["positive"] for r in rows], [r["null"] for r in rows]
        result = paired_counts(positive, null)
        total_improved += result["improved"]
        total_regressed += result["regressed"]
        for group in sorted({r["seed_index"] for r in rows}):
            block = [int(r["positive"]) - int(r["null"]) for r in rows if r["seed_index"] == group]
            probabilities = np.bincount(np.asarray(block) + 1, minlength=3) / len(block)
            sampled = rng.multinomial(len(block), probabilities, size=bootstrap_replicates)
            bootstrap += (sampled[:, 2] - sampled[:, 0]) / (len(tasks) * expected_per_task)
        per_task[task] = result
    adjusted = holm_adjust({task: value["two_sided_p"] for task, value in per_task.items()})
    for task in tasks:
        per_task[task]["holm_p"] = adjusted[task]
    macro = float(np.mean([r["delta"] for r in per_task.values()]))
    interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    p = exact_p(total_improved, total_regressed)
    unadapted = [per_task[t]["delta"] for t in tasks if t not in registry_tasks]
    guard_ok = not unadapted or (min(unadapted) >= -0.05 and np.mean(unadapted) >= -0.002)
    return {
        "schema_version": 1, "estimand": "equal_weight_mean_on_fixed_task_suite",
        "tasks": per_task, "task_count": len(tasks), "episodes_per_condition": len(tasks) * expected_per_task,
        "null_macro_success": float(np.mean([r["null_success"] / expected_per_task for r in per_task.values()])),
        "positive_macro_success": float(np.mean([r["positive_success"] / expected_per_task for r in per_task.values()])),
        "macro_delta": macro, "paired_bootstrap_95ci": interval, "two_sided_exact_p": p,
        "total_improved": total_improved, "total_regressed": total_regressed,
        "registry_tasks": sorted(registry_tasks), "unadapted_guard_passed": bool(guard_ok),
        "statistical_uplift": bool(macro > 0 and interval[0] > 0 and p < 0.05),
        "passed": bool(macro > 0 and interval[0] > 0 and p < 0.05 and guard_ok),
        "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": bootstrap_seed,
        "scope_warning": "Does not imply every task improved or generalization to unseen tasks.",
    }


def power_sensitivity(*, simulations=20000, seed=20260908):
    """Conditional final exact-test power, NOT whole-pipeline or guard-gate power."""
    rng = np.random.default_rng(seed)
    cache, rows = {}, []
    for adapted in (1, 2, 4):
        for delta in (0.10, 0.15, 0.20):
            for guard_discordance in (0.0, 0.005, 0.01):
                # 1% regressions on adapted tasks; remaining discordance is improvement.
                a = rng.multinomial(60 * adapted, [delta + 0.01, 0.01, 0.98 - delta], size=simulations)
                b = rng.multinomial(60 * (50 - adapted), [guard_discordance / 2] * 2 + [1 - guard_discordance], size=simulations)
                counts = a[:, :2] + b[:, :2]
                passed = 0
                for improved, regressed in counts:
                    key = (int(improved), int(regressed))
                    if key not in cache:
                        cache[key] = exact_p(*key) < 0.05 and improved > regressed
                    passed += cache[key]
                rows.append({"adapted_tasks": adapted, "task_uplift": delta,
                             "unadapted_discordance_rate": guard_discordance,
                             "suite_uplift": adapted * delta / 50,
                             "exact_test_power": float(passed / simulations)})
    return {"simulations": simulations, "seed": seed, "rows": rows,
            "warning": "Conditional exact-test sensitivity; not a significance guarantee or power of all study gates."}
