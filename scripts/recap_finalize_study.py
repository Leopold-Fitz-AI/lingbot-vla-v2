#!/usr/bin/env python3
"""Terminal-only audit/report observer. NEVER launches or retries a policy job.

Run outside the frozen code directory, with PYTHONPATH set to STUDY/code.
Statistics are recomputed using that unchanged snapshot, only AFTER the
controller has completed the entire final stage. No interim success peeking.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path


COMPLETE_STATES = {"significant_uplift", "not_proven"}
STOP_STATES = {"infrastructure_or_protocol_failure", "no_validated_candidate"}


def read_json(path):
    return json.loads(path.read_text())


def require_complete(root):
    status = read_json(root / "status.json")
    if status.get("stage") != "complete" or status.get("state") not in COMPLETE_STATES:
        raise RuntimeError("Final analysis refused: the controller has not completed the final stage")
    return status


def progress(root):
    """Reads status/job metadata, never episode outcomes or final result files."""
    status = read_json(root / "status.json")
    counts = {"primary_validated": 0, "negative_validated": 0, "recorded_not_validated": 0}
    for job in (root / "evaluations/final").glob("*/*/*/job.json"):
        spec = read_json(job)
        done = job.parent / "done.json"
        if done.exists():
            key = "negative_validated" if spec["condition"] == "negative" else "primary_validated"
            counts[key] += len(read_json(done)["manifests"])
        else:
            # Counts paths only. Repeated attempts are overhead, not new samples.
            for task in spec["tasks"]:
                attempts = [p for p in (job.parent / "rollouts").glob("attempt-*") if (p / task).is_dir()]
                if attempts:
                    latest = max(attempts, key=lambda p: int(p.name.split("-")[-1]))
                    counts["recorded_not_validated"] += len(list((latest / task).rglob("manifest.json")))
    return {"stage": status["stage"], "state": status["state"], **counts,
            "final_opened": (root / "final_opened.json").exists(),
            "final_preflight_locked": {str(i): len(list((root / "cohorts" / f"s{i}").glob("*/lock.json")))
                                       for i in (50, 51, 52)}}


def controller_alive_or_terminal(root):
    status = read_json(root / "status.json")
    if status["state"] in COMPLETE_STATES | STOP_STATES:
        return True
    try:
        os.kill(status["pid"], 0)
        cmdline = Path(f"/proc/{status['pid']}/cmdline")
        if cmdline.exists() and str(root).encode() not in cmdline.read_bytes():
            raise ProcessLookupError("Controller PID was reused")
        return True
    except (ProcessLookupError, FileNotFoundError):
        # The controller may publish completion and exit between our two reads.
        return read_json(root / "status.json")["state"] in COMPLETE_STATES | STOP_STATES


def make_pairs(plan, collected, registry_tasks):
    if set(collected) != {"positive", "null", "negative"}:
        raise ValueError("Missing final condition")
    tasks, indices = plan["all_tasks"], plan["final"]["seed_indices"]
    count = plan["final"]["episodes_per_index"]
    for condition, members in collected.items():
        expected_tasks = set(registry_tasks) if condition == "negative" else set(tasks)
        if set(members) != expected_tasks:
            raise ValueError("Missing/unexpected final tasks")
        for task, rows in members.items():
            if len(rows) != count * len(indices):
                raise ValueError(f"Incomplete final cohort: {condition}/{task}")
            groups = [seed // 100000 - 1 for seed in rows]
            if set(groups) != set(indices) or any(groups.count(i) != count for i in indices):
                raise ValueError("Incomplete/changed task-by-index strata")
            if set(rows) != set(collected["null"][task]):
                raise ValueError("Unpaired final seeds")
    return {task: [{"seed": seed, "seed_index": seed // 100000 - 1,
                    "positive": collected["positive"][task][seed][1]["success"],
                    "null": collected["null"][task][seed][1]["success"]}
                   for seed in sorted(collected["null"][task])] for task in tasks}


def markdown(report, root):
    ci = report["paired_bootstrap_95ci"]
    text = ["# Fixed 50-task final study", "", f"Study: `{root}`", "",
            "**All frozen success criteria passed.**" if report["passed"] else "**Suite improvement NOT established under all frozen criteria.**",
            "", f"- Primary episodes: {2 * report['episodes_per_condition']} (paired, both policies actually executed).",
            f"- Null macro success: {100 * report['null_macro_success']:.4f}%.",
            f"- RECAP macro success: {100 * report['positive_macro_success']:.4f}%.",
            f"- Equal-task macro change: {100 * report['macro_delta']:+.4f} percentage points.",
            f"- Paired 95% bootstrap CI: [{100 * ci[0]:+.4f}, {100 * ci[1]:+.4f}] pp.",
            f"- Exact two-sided paired p: {report['two_sided_exact_p']:.8g}.",
            f"- Improved/regressed pairs: {report['total_improved']}/{report['total_regressed']}.",
            f"- Unadapted regression guards passed: {report['unadapted_guard_passed']}.",
            f"- Adapted tasks: {', '.join(report['registry_tasks']) or 'none'}.",
            "", "This is a fixed-suite mean claim, NOT improvement on every task or unseen-task generalization.",
            "Historical 4496/5000 (89.92%) is inventory only, not this experiment's comparator.",
            "No production registry promotion. No extra samples or outcome-dependent retries.", "",
            "## All task contributions", "",
            "| Task | Adapted | Null | RECAP | Delta (pp) | Suite contribution (pp) | Wins / losses | Holm p |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for task, row in sorted(report["tasks"].items()):
        text.append(f"| {task} | {'yes' if task in report['registry_tasks'] else 'no'} | "
                    f"{row['null_success']}/{row['episodes']} | {row['positive_success']}/{row['episodes']} | "
                    f"{100 * row['delta']:+.3f} | {100 * row['delta'] / report['task_count']:+.4f} | "
                    f"{row['improved']} / {row['regressed']} | {row['holm_p']:.6g} |")
    text += ["", "## Secondary signed-Negative diagnostic", ""]
    for task, row in sorted(report["secondary_negative"].items()):
        text.append(f"- {task}: Negative {row['positive_success']}/{row['episodes']}; "
                    f"Null {row['null_success']}/{row['episodes']}. Not a primary endpoint or selection rule.")
    return "\n".join(text) + "\n"


def audit(root, engine, evaluation):
    status = require_complete(root)  # MUST precede reading ANY policy outcomes.
    engine.verify()
    plan = engine.plan
    if (len(plan["all_tasks"]) != 50 or plan["final"] != {"seed_indices": [50, 51, 52], "episodes_per_index": 20}
            or plan["bootstrap_replicates"] != 20000 or plan["bootstrap_seed"] != 20260908):
        raise ValueError("Observer supports only the frozen 50-task design")
    sha = evaluation.sha256
    registry_path = root / "registries/final.json"
    registry = read_json(registry_path)["tasks"]
    freeze = read_json(root / "registry_freeze.json")
    if (freeze["registry_sha256"] != sha(registry_path)
            or freeze["plan_sha256"] != sha(root / "plan.json")
            or freeze["confirmation_sha256"] != sha(root / "confirmation.json")
            or read_json(root / "final_opened.json")["registry_sha256"] != sha(registry_path)):
        raise ValueError("Registry/admission/opening provenance mismatch")
    admitted = {task for task, row in read_json(root / "confirmation.json").items() if row["passed"]}
    if set(registry) != admitted:
        raise ValueError("Final registry differs from independent admission")
    collected = {role: {} for role in ("positive", "null", "negative")}
    verified = {}

    def verify_mirror(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Local evidence checksum mismatch: {path}")
        mirror = engine.public / path.relative_to(root)
        if not mirror.is_file() or sha(mirror) != digest:
            raise ValueError(f"Persistent evidence read-back mismatch: {mirror}")
        verified[str(path.relative_to(root))] = digest

    for job in sorted((root / "evaluations/final").glob("*/*/*/job.json")):
        spec = read_json(job)
        role, index = spec["condition"], spec["seed_index"]
        expected_path = root / "registries" / ("empty.json" if role == "null" else "final.json")
        if (role not in collected or index not in (50, 51, 52) or spec["count"] != 20
                or spec["code_sha256"] != plan["code_sha256"] or spec["registry_path"] != str(expected_path)
                or any(plan["task_gpu"][t] != spec["gpu"] for t in spec["tasks"])):
            raise ValueError("Final job changed frozen protocol/assignment")
        rows = engine.read_job(job.parent)
        done_path = job.parent / "done.json"
        done = read_json(done_path)
        verify_mirror(done_path, read_json(job.parent / "persisted.json")["done_sha256"])
        referenced = set()
        for task, samples in rows.items():
            dest = collected[role].setdefault(task, {})
            if dest.keys() & samples.keys() or any(s // 100000 - 1 != index for s in samples):
                raise ValueError("Duplicate/unplanned final state")
            dest.update(samples)
            for path, row in samples.values():
                referenced.update(str(p.relative_to(job.parent)) for p in
                                  [path, *(path.parent / step["file"] for step in row["steps"])])
        if referenced != {r["path"] for r in done["files"]} or len(done["files"]) != len(referenced):
            raise ValueError("Incomplete/duplicate rollout checksum inventory")
        for record in done["files"]:
            verify_mirror(job.parent / record["path"], record["sha256"])
        for name in ("job.json", "cohorts.json", "seeds.json", "instructions.json", "persisted.json"):
            verify_mirror(job.parent / name)
    pairs = make_pairs(plan, collected, registry)
    engine.paired(collected["positive"], collected["null"])
    report = evaluation.final_statistics(pairs, tasks=plan["all_tasks"], registry_tasks=registry,
                                        expected_per_task=60, bootstrap_replicates=20000, bootstrap_seed=20260908)
    report["secondary_negative"] = engine.paired(collected["negative"], {t: collected["null"][t] for t in registry})
    report.update(registry_sha256=sha(registry_path), plan_sha256=sha(root / "plan.json"))
    if pairs != read_json(root / "final_pairs.json") or report != read_json(root / "final_result.json"):
        raise ValueError("Frozen statistics do not reproduce from complete authoritative manifests")
    if (status["state"] == "significant_uplift") != report["passed"]:
        raise ValueError("Controller terminal claim disagrees with all frozen gates")
    for name in ("plan.json", "registry_freeze.json", "confirmation.json", "registries/final.json",
                 "final_pairs.json", "final_result.json"):
        verify_mirror(root / name)
    for path in (root / "cohorts").glob("s5[012]/*/preflight.json"):
        verify_mirror(path, read_json(path.with_name("lock.json"))["sha256"])
        verify_mirror(path.with_name("lock.json"))
    return report, {"verified_files": verified, "reproduced_frozen_statistics": True,
                    "primary_policy_episodes": 6000, "secondary_policy_episodes": 60 * len(registry),
                    "no_policy_execution_or_retries": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root = args.study.resolve()
    # PYTHONPATH must resolve these imports to the existing frozen snapshot.
    from lingbotvla.recap import evaluation
    from scripts import recap_confirmatory_study as frozen
    if (Path(frozen.__file__).resolve() != root / "code/scripts/recap_confirmatory_study.py"
            or Path(evaluation.__file__).resolve() != root / "code/lingbotvla/recap/evaluation.py"):
        raise RuntimeError("Set PYTHONPATH=STUDY/code; never substitute an analysis implementation")
    if args.poll_seconds < 1:
        parser.error("poll interval must be positive")
    out = root / "reports"
    out.mkdir(exist_ok=True)
    engine = frozen.Study(root)
    identity = {"observer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "plan_sha256": evaluation.sha256(root / "plan.json")}
    with (out / "completion_observer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            last = None
            while True:
                current = progress(root)
                if current != last:
                    print(json.dumps(current, sort_keys=True), flush=True)
                    last = current
                if current["state"] in COMPLETE_STATES:
                    break
                if current["state"] in STOP_STATES:
                    raise RuntimeError(f"Study stopped without a complete final test: {read_json(root / 'status.json')}")
                if not args.watch:
                    require_complete(root)
                if not controller_alive_or_terminal(root):
                    raise RuntimeError("Controller is absent with a nonterminal status; manual audit required")
                time.sleep(args.poll_seconds)
            report, evidence = audit(root, engine, evaluation)
            evidence.update(identity)
            evaluation.write_json(out / "completion_audit_v1.json", evidence, immutable=True)
            path = out / "completion_report_v1.md"
            content = markdown(report, root)
            if path.exists() and path.read_text() != content:
                raise ValueError("Immutable completion report changed")
            path.write_text(content)
            if path.read_text() != content:
                raise IOError("Completion report read-back failed")
            engine.persist([out / "completion_audit_v1.json", path])
            print("COMPLETION_AUDIT_PASSED", report["passed"], str(path), flush=True)
        except Exception as error:
            path = out / "completion_observer_failure_v1.json"
            evaluation.write_json(path, {**identity, "error": repr(error), "controller_status": read_json(root / "status.json"),
                                         "no_final_estimate_from_incomplete_data": True}, immutable=True)
            engine.persist([path])
            raise


if __name__ == "__main__":
    main()
