import concurrent.futures
import copy
import threading

import pytest

from lingbotvla.recap.evaluation import sha256, write_json
from scripts.recap_confirmatory_study import Study, inherit_cohorts


def test_parallel_reports_failure_before_waiting_for_other_workers():
    study = Study.__new__(Study)
    study.stop_requested = threading.Event()
    started, reported = threading.Event(), threading.Event()
    statuses = []

    def status(stage, state, **extra):
        statuses.append((stage, state, extra))
        reported.set()

    study.status = status

    def slower():
        started.set()
        assert reported.wait(3), "failure was hidden until another worker finished"
        assert study.stop_requested.is_set()
        return "drained"

    def failed():
        assert started.wait(3)
        raise ValueError("frozen seed failed")

    with pytest.raises(ValueError, match="frozen seed failed"):
        study.parallel([(slower, ()), (failed, ())])
    assert statuses[0][1] == "infrastructure_or_protocol_failure_waiting_for_workers"
    with pytest.raises(concurrent.futures.CancelledError):
        study.evaluate(None, None, None, None, None, None, None, None)
    with pytest.raises(concurrent.futures.CancelledError):
        study.execute([], cwd=None, env=None, log=None)


def test_parallel_preserves_declared_order():
    study = Study.__new__(Study)
    assert study.parallel([(lambda n: n * 2, (n,)) for n in range(6)]) == list(range(0, 12, 2))


@pytest.fixture
def predecessor(tmp_path):
    previous, destination = tmp_path / "old", tmp_path / "new"
    previous.mkdir()
    plan = {key: {} for key in ("base_files", "task_gpu", "target_gpu", "confirm", "final")}
    plan.update(all_tasks=["bell"], target_tasks=["bell"], gpus=[4, 5, 6, 7],
                wrapper_sha256="wrapper", control={"episodes_per_index": 2}, screen={"episodes_per_index": 2},
                confirmation_family=["bell"], confirmation_holm_alpha=0.05, confirmation_min_delta=0.05,
                bootstrap_replicates=20000, bootstrap_seed=20260908,
                protocol={"policy_seed": 900},
                candidates={"bell": [{"name": "positive_all", "sha256": "adapter", "path": "old/adapter"}]})
    write_json(previous / "plan.json", plan)
    write_json(previous / "plan_lock.json", {"sha256": sha256(previous / "plan.json")})
    write_json(previous / "status.json", {"state": "infrastructure_or_protocol_failure"})
    for index in (32, 42):
        folder = previous / "cohorts" / f"s{index}" / "bell"
        write_json(folder / "preflight.json", {"task": "bell", "episodes": 2, "policy_rollouts": 0})
        write_json(folder / "lock.json", {"sha256": sha256(folder / "preflight.json")})
    write_json(previous / "evaluations/screen/outcomes.json", {"never": "reuse"})
    new_plan = copy.deepcopy(plan)
    new_plan["protocol"]["eligibility_protocol"] = "locked-expert-preflight-v1"
    new_plan["candidates"]["bell"][0]["path"] = "new/adapter"
    return previous, destination, new_plan


def test_recovery_preserves_cohorts_but_never_imports_policy_results(predecessor):
    previous, destination, plan = predecessor
    record = inherit_cohorts(destination, previous, plan)
    assert record["policy_results_reused"] is False
    assert len(record["cohorts"]) == 2
    for path in (previous / "cohorts").rglob("*.json"):
        assert path.read_bytes() == (destination / path.relative_to(previous)).read_bytes()
    assert not (destination / "evaluations").exists()


@pytest.mark.parametrize("changed", ["weights", "policy", "threshold", "late_cohort", "cohort_hash", "running"])
def test_recovery_rejects_changed_design_or_unfrozen_data(predecessor, changed):
    previous, destination, plan = predecessor
    if changed == "weights":
        plan["candidates"]["bell"][0]["sha256"] = "different"
    if changed == "policy":
        plan["protocol"]["policy_seed"] = 901
    if changed == "threshold":
        plan["confirmation_min_delta"] = 0.01
    if changed == "late_cohort":
        (previous / "cohorts/s43").mkdir()
    if changed == "cohort_hash":
        write_json(previous / "cohorts/s42/bell/preflight.json", {"corrupted": True})
    if changed == "running":
        write_json(previous / "status.json", {"state": "running"})
    with pytest.raises(ValueError):
        inherit_cohorts(destination, previous, plan)


def test_progress_does_not_label_a_failed_job_as_healthy_running(tmp_path):
    from scripts.recap_study_status import study_status
    write_json(tmp_path / "status.json", {"state": "running", "stage": "screen_42"})
    folder = tmp_path / "evaluations/screen/s42/positive/gpu4"
    write_json(folder / "job.json", {"phase": "screen", "tasks": ["bell"], "count": 20})
    write_json(folder / "failure.json", {"error": "initial state mismatch"})
    result = study_status(tmp_path)
    assert result["state"] == "failure_detected_waiting_for_workers"
    assert result["stages"]["screen"]["failed_jobs"] == 1
    assert result["final_seeds_opened"] is False
