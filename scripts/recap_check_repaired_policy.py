#!/usr/bin/env python3
"""One bounded, exploratory closed-loop check of existing P0 models versus base.

Reuses the audited cohort/evaluation executor. No training, candidate selection,
Negative arm, pilot collector, final-suite access, or automatic next experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from lingbotvla.recap.evaluation import holm_adjust, sha256, write_json
from scripts.recap_confirmatory_study import Study, code_hash, read_json


TASKS = ("hanging_mug", "place_can_basket", "stack_bowls_three")
COUNT = 40
INDEX = 120
WINDOWS = {"hanging_mug": (0, -1), "place_can_basket": (1, 1), "stack_bowls_three": (1, 1)}


def resources():
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    used = {int(a): int(b) for a, b in (line.split(",") for line in text.strip().splitlines())}
    if any(used[gpu] > 1024 for gpu in (5, 6, 7)):
        raise RuntimeError("Authorized GPUs 5–7 are not idle; no launch")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    uuids = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    for line in uuids.strip().splitlines():
        index, uuid = line.split(",")
        if int(index) in (5, 6, 7) and uuid.strip() in processes:
            raise RuntimeError("Authorized GPU has a compute process; no launch")
    return {"checked_at_unix": time.time(), "memory_mib": used, "compute_processes": processes}


def initialize(root, source, seed_audit):
    if root.exists():
        raise FileExistsError("Never overwrite/restart an existing check")
    ledger = read_json(seed_audit)
    if ledger.get("seed_index") != INDEX or ledger.get("unused") is not True:
        raise ValueError("Unused seed-index audit required")
    old = Path("/dev/shm/recap_confirmatory_20260908_locked_cohorts")
    # Read only the old design, NOT old cohort inputs or policy results.
    previous = read_json(old / "plan.json")
    p0 = Path("/dev/shm/recap_ablation_20260910/study")
    report = read_json(p0 / "reports/initialization_ablation.json")
    if report.get("complete") is not True or report.get("formal_optimizer_seed") != 2101:
        raise ValueError("Completed formally seeded P0 report required")
    root.mkdir(parents=True)
    shutil.copytree(source, root / "code", ignore=shutil.ignore_patterns("__pycache__", ".git", ".remote_audit"))
    model = root / "model"
    hf = model / "checkpoints/global_step_50000/hf_ckpt"
    hf.parent.mkdir(parents=True)
    hf.symlink_to(previous["base_checkpoint"], target_is_directory=True)
    shutil.copy2(old / "model/lingbotvla_cli.yaml", model / "lingbotvla_cli.yaml")
    shutil.copy2(seed_audit, root / "seed_audit.json")
    candidates, entries = {}, {}
    for task in TASKS:
        folder = p0 / task / "deployment_s2101/orthogonal_matched_v1"
        training = read_json(folder / "training.json")
        src = folder / "recap_adapter.safetensors"
        if training["artifact"]["sha256"] != sha256(src) or training["base_tensors_unchanged"] is not True:
            raise ValueError("Changed/unverified P0 adapter")
        dst = root / "artifacts" / task / src.name
        dst.parent.mkdir(parents=True)
        shutil.copy2(src, dst)
        shutil.copy2(folder / "training.json", dst.parent / "training.json")
        start, decisions = WINDOWS[task]
        entries[task] = {
            "path": str(dst),
            "sha256": sha256(dst),
            "condition_start_decision": start,
            "condition_decisions": decisions,
        }
        candidates[task] = [entries[task]]
    plan = {
        key: copy.deepcopy(previous[key])
        for key in (
            "project",
            "protocol",
            "inference_env",
            "sim_env",
            "conda_sh",
            "eval_workdir",
            "base_checkpoint",
            "base_files",
        )
    }
    plan.update(
        schema_version=1,
        created_at_unix=time.time(),
        study=str(root),
        all_tasks=list(TASKS),
        target_tasks=list(TASKS),
        candidates=candidates,
        gpus=[5, 6, 7],
        task_gpu=dict(zip(TASKS, (5, 6, 7))),
        model_path=str(hf),
        code_sha256=code_hash(root / "code"),
        wrapper_sha256=sha256(model / "lingbotvla_cli.yaml"),
        preregistration_sha256=sha256(root / "code/docs/recap_confirmatory_protocol.md"),
        check_protocol_sha256=sha256(root / "code/docs/recap_repaired_policy_check_20260911.md"),
        seed_audit_sha256=sha256(root / "seed_audit.json"),
        seed_index=INDEX,
        episodes_per_task_arm=COUNT,
        maximum_policy_episodes_excluding_infra_retries=240,
        formal_optimizer_seed=2101,
        backend="deployment",
        initializer="orthogonal_matched_v1",
        condition="positive",
        order=["null", "repaired"],
        scope="exploratory_three_task_existing_model_check_not_confirmatory_or_factor_attribution",
        bootstrap_seed=20260911,
        bootstrap_replicates=20000,
        prioritization="three-task macro delta >=5pp AND no task net regression; not an efficacy proof",
        automatic_next_stage=False,
        training_steps=0,
    )
    write_json(root / "plan.json", plan, immutable=True)
    write_json(root / "plan_lock.json", {"sha256": sha256(root / "plan.json")}, immutable=True)
    write_json(root / "registries/empty.json", {"schema_version": 1, "tasks": {}}, immutable=True)
    write_json(root / "registries/repaired.json", {"schema_version": 1, "tasks": entries}, immutable=True)
    return plan


def summarize(pairs, counts):
    if set(pairs) != set(TASKS) or any(len(pairs[t]) != COUNT for t in TASKS):
        raise ValueError("Complete three-task paired cohort required")
    delta = np.array([pairs[t] for t in TASKS], dtype=np.float64)
    if not np.isin(delta, (-1, 0, 1)).all():
        raise ValueError("Invalid paired outcome differences")
    rng = np.random.default_rng(20260911)
    draws = np.zeros(20000)
    for row in delta:
        draws += row[rng.integers(0, COUNT, size=(20000, COUNT))].mean(axis=1) / 3
    macro = float(delta.mean())
    return {
        "complete": True,
        "scope": "Exploratory; no automatic promotion or next experiment",
        "policy_episodes": 240,
        "independent_paired_states": 120,
        "per_task": counts,
        "macro_delta_pp": 100 * macro,
        "macro_paired_bootstrap_95ci_pp": (100 * np.quantile(draws, [0.025, 0.975])).tolist(),
        "task_holm_p": holm_adjust({t: counts[t]["two_sided_p"] for t in TASKS}),
        "worth_further_investigation": bool(delta.sum() >= 6 and all(row.sum() >= 0 for row in delta)),
        "interpretation": "Priority flag is not proof; a false flag does not establish no benefit. N=40/task may be inconclusive.",
    }


def run(root):
    study = Study(root)
    if Path(__file__).resolve() != (study.code / "scripts/recap_check_repaired_policy.py").resolve():
        raise ValueError("Run only the frozen check source")
    study.verify()
    for file, key in (
        ("seed_audit.json", "seed_audit_sha256"),
        ("code/docs/recap_repaired_policy_check_20260911.md", "check_protocol_sha256"),
    ):
        if sha256(root / file) != study.plan[key]:
            raise ValueError("Frozen check design changed")
    write_json(root / "resource_check.json", resources(), immutable=True)
    # Permanent one-shot marker. No supervisor retry or outcome-dependent resume.
    with (root / "launch_once.json").open("x") as f:
        json.dump({"pid": os.getpid(), "started_at_unix": time.time()}, f)
        f.flush()
        os.fsync(f.fileno())
    try:
        study.persist([p for p in sorted(root.rglob("*")) if p.is_file() and "code" not in p.relative_to(root).parts])
        study.sync_sim()
        study.status("preflight")
        study.cohorts(TASKS, INDEX, COUNT, study.plan["task_gpu"])
        study.status("paired_evaluation")

        def worker(task):
            args = ("check", INDEX)
            gpu = study.plan["task_gpu"][task]
            null = study.evaluate(*args, "null", [task], COUNT, gpu, study.empty, "null")
            repaired = study.evaluate(
                *args, "repaired", [task], COUNT, gpu, root / "registries/repaired.json", "positive"
            )
            return repaired, null

        completed = study.parallel([(worker, (t,)) for t in TASKS])
        positive, null = {}, {}
        for a, b in completed:
            positive.update(a)
            null.update(b)
        counts = study.paired(positive, null)
        pairs = {
            t: [int(positive[t][s][1]["success"]) - int(null[t][s][1]["success"]) for s in sorted(positive[t])]
            for t in TASKS
        }
        result = summarize(pairs, counts)
        write_json(
            root / "paired_outcomes.json",
            {
                t: [
                    {"seed": s, "positive": positive[t][s][1]["success"], "null": null[t][s][1]["success"]}
                    for s in sorted(positive[t])
                ]
                for t in TASKS
            },
            immutable=True,
        )
        write_json(root / "result.json", result, immutable=True)
        study.persist([root / "paired_outcomes.json", root / "result.json"])
        study.status("complete", "complete", result="result.json")
    except BaseException as error:
        study.stop_requested.set()
        write_json(root / "failure.json", {"error": repr(error), "time": time.time()}, immutable=True)
        study.persist([root / "failure.json"])
        study.status("stopped", "infrastructure_or_protocol_failure", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--seed-audit", type=Path)
    args = parser.parse_args()
    if args.initialize:
        if args.source is None or args.seed_audit is None:
            parser.error("Initialization requires source and seed audit")
        initialize(args.root, args.source, args.seed_audit)
    else:
        run(args.root)


if __name__ == "__main__":
    main()
