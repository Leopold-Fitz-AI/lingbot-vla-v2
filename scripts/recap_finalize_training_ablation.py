#!/usr/bin/env python3
"""Terminal-only read-back audit for P0; never starts/retries training or policies."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from lingbotvla.recap.ablation import BACKENDS, INITIALIZATIONS, SEEDS, TASKS, summarize_pairs
from lingbotvla.recap.evaluation import sha256, write_json


def load(p):
    return json.loads(p.read_text())


def audit(root):
    import torch
    from safetensors.torch import load_file

    plan = load(root / "plan.json")
    if sha256(root / "plan.json") != load(root / "lock.json")["plan_sha256"]:
        raise ValueError("Changed plan")
    if (root / "stop.json").exists() or any((root / t / "failure.json").exists() for t in TASKS):
        raise ValueError("Incomplete/failed ablation must not be analyzed as a successful cohort")
    if any(not (root / t / "done.json").exists() for t in TASKS):
        raise ValueError("Ablation is not complete")
    source = root.parent / "code"
    for p, h in plan["source_sha256"].items():
        if sha256(source / p) != h:
            raise ValueError(f"Frozen code changed: {p}")
    for p, h in plan["evidence_sha256"].items():
        if sha256(Path(p)) != h:
            raise ValueError(f"Input changed: {p}")
    expected_base = load(Path(plan["parity_report"]))["base_tensor_sha256"]
    if len(expected_base) != 1708:
        raise ValueError("Unexpected base tensor inventory")
    cells = {}
    for task in TASKS:
        done = load(root / task / "done.json")
        if done["cells"] != 12 or done["optimizer_steps"] != 600 or done["base_tensor_sha256"] != expected_base:
            raise ValueError("Incomplete/mutated base training result")
        rows = load(root / task / "diagnostic_rows.json")
        if {k: summarize_pairs(v) for k, v in rows.items()} != done["diagnostics"]:
            raise ValueError("Consumed-state diagnostics do not reproduce")
        for seed in SEEDS:
            common = None
            for backend in BACKENDS:
                for init in INITIALIZATIONS:
                    key = f"{backend}_s{seed}/{init}"
                    folder = root / task / key
                    record = load(folder / "training.json")
                    cell = next(
                        c
                        for c in plan["cells"]
                        if c["task"] == task
                        and c["optimizer_seed"] == seed
                        and c["backend"] == backend
                        and c["initialization"] == init
                    )
                    if record["effective_cell"] != cell or record["steps"] != 50 or len(record["updates"]) != 50:
                        raise ValueError("Training cell changed")
                    if not (
                        record["base_tensors_unchanged"]
                        and record["verified_base_tensors"] == 1708
                        and record["cache_loss_exact"]
                        and record["cache_gradient_gate_exact"]
                    ):
                        raise ValueError("Missing real optimization integrity gates")
                    if [r["step"] for r in record["updates"]] != list(range(1, 51)):
                        raise ValueError("Repeated/missing optimizer step")
                    draws = record["draws"]
                    if len(draws) != 200 or [d["row"] for d in draws] != sum(
                        plan["tasks"][task]["schedules"][str(seed)], []
                    ):
                        raise ValueError("Training sample schedule changed")
                    if common is not None and draws != common:
                        raise ValueError("Unmatched factorial inputs")
                    common = draws
                    artifact = folder / "recap_adapter.safetensors"
                    if sha256(artifact) != record["artifact"]["sha256"]:
                        raise ValueError("Adapter hash changed")
                    tensors = load_file(str(artifact))
                    if set(tensors) != {"model.recap_velocity_lora_a", "model.recap_velocity_lora_b"}:
                        raise ValueError("Unexpected exported tensors")
                    a, b = tensors["model.recap_velocity_lora_a"], tensors["model.recap_velocity_lora_b"]
                    if (
                        a.dtype != torch.float32
                        or b.dtype != torch.float32
                        or not torch.isfinite(a).all()
                        or not torch.isfinite(b).all()
                    ):
                        raise ValueError("Nonfinite/wrong dtype adapter")
                    singular_a = torch.linalg.svdvals(a[1].double()).numpy()
                    singular_ba = torch.linalg.svdvals(b[1].double() @ a[1].double()).numpy()

                    def energy(s):
                        return float(np.sum(s[:2] ** 2) / np.sum(s**2)) if np.sum(s**2) > 0 else None

                    cells[f"{task}/{key}"] = {
                        "artifact_sha256": sha256(artifact),
                        "task": task,
                        "seed": seed,
                        "backend": backend,
                        "initialization": init,
                        "a_singular_values": singular_a.tolist(),
                        "ba_singular_values": singular_ba.tolist(),
                        "a_top_two_energy": energy(singular_a),
                        "ba_top_two_energy": energy(singular_ba),
                        "ba_frobenius": float(np.linalg.norm(singular_ba)),
                        "diagnostics": done["diagnostics"][key],
                    }
    if len(cells) != 36:
        raise ValueError("All 36 cells are required")
    differences = {}
    for task in TASKS:
        differences[task] = {}
        for split in ("train", "holdout"):
            measurements = {}
            for backend in BACKENDS:
                for init in INITIALIZATIONS:
                    scores = [
                        cells[f"{task}/{backend}_s{seed}/{init}"]["diagnostics"][split]["surrogate_margin_mean"]
                        for seed in SEEDS
                    ]
                    measurements[f"{backend}/{init}"] = {
                        "seeds": list(SEEDS),
                        "values": scores,
                        "mean": float(np.mean(scores)),
                    }
            differences[task][split] = measurements
    return {
        "complete": True,
        "training_runs": 36,
        "optimizer_updates": 1800,
        "policy_episodes": 0,
        "base_tensors_unchanged": True,
        "base_tensor_count": 1708,
        "formal_optimizer_seed": plan["formal_optimizer_seed"],
        "plan_sha256": sha256(root / "plan.json"),
        "cells": cells,
        "all_seed_surrogate_margins": differences,
        "scope": "All factorial cells, descriptive consumed-state diagnostics only; no success-rate/p-value claim or promotion",
    }


def persist(root, mirror):
    inventory = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix == ".lock":
            continue
        rel = p.relative_to(root)
        if str(rel) == "reports/persistence.json":
            continue
        h = sha256(p)
        target = mirror / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != h:
                raise ValueError(f"Persistent evidence conflicts: {rel}")
        else:
            shutil.copy2(p, target)
            with target.open("rb") as f:
                os.fsync(f.fileno())
        if sha256(target) != h:
            raise OSError(f"Read-back failed: {rel}")
        inventory[str(rel)] = h
    return inventory


def workers_alive(root, proc=Path("/proc")):
    for task, row in load(root / "worker_pids.json").items():
        if (root / task / "done.json").exists() or (root / task / "failure.json").exists():
            continue
        try:
            path = proc / str(row["pid"])
            command = (path / "cmdline").read_bytes()
            ticks = (path / "stat").read_text().split(") ", 1)[1].split()[19]
            if str(root).encode() not in command or ticks != row["start_ticks"]:
                return False
        except (FileNotFoundError, ProcessLookupError):
            if not ((root / task / "done.json").exists() or (root / task / "failure.json").exists()):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    out = args.root / "reports"
    out.mkdir(exist_ok=True)
    with (out / "observer.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            failed = (args.root / "stop.json").exists() or any(
                (args.root / t / "failure.json").exists() for t in TASKS
            )
            terminal = all(
                (args.root / t / "done.json").exists() or (args.root / t / "failure.json").exists() for t in TASKS
            )
            if terminal:
                break
            if not args.watch:
                raise RuntimeError("Refusing incomplete ablation analysis")
            if not workers_alive(args.root):
                error = "Worker disappeared without terminal evidence; manual audit required, no retries"
                write_json(out / "observer_failure.json", {"error": error}, immutable=True)
                raise RuntimeError(error)
            print(json.dumps({"waiting_for_workers": True, "failure_detected": failed}), flush=True)
            time.sleep(30)
        try:
            if failed:
                report = {
                    "complete": False,
                    "state": "training_or_protocol_failure",
                    "retries": 0,
                    "failures": {
                        t: load(args.root / t / "failure.json")
                        for t in TASKS
                        if (args.root / t / "failure.json").exists()
                    },
                }
                write_json(out / "failure_report.json", report, immutable=True)
            else:
                report = audit(args.root)
                write_json(out / "initialization_ablation.json", report, immutable=True)
            inventory = persist(args.root, args.mirror)
            write_json(
                out / "persistence.json",
                {"verified": True, "mirror": str(args.mirror), "files_sha256": inventory},
                immutable=True,
            )
            target = args.mirror / "reports/persistence.json"
            shutil.copy2(out / "persistence.json", target)
            if sha256(target) != sha256(out / "persistence.json"):
                raise OSError("Persistence receipt changed")
            print(
                json.dumps({"complete": report["complete"], "read_back_files": len(inventory), "policy_episodes": 0}),
                flush=True,
            )
        except Exception as exc:
            write_json(
                out / "observer_failure.json", {"error": repr(exc), "no_training_restart": True}, immutable=True
            )
            raise


if __name__ == "__main__":
    main()
