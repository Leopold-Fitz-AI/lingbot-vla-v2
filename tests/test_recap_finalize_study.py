from pathlib import Path
from types import SimpleNamespace

import pytest

from lingbotvla.recap.evaluation import final_statistics, write_json
from scripts.recap_finalize_study import (
    audit,
    controller_alive_or_terminal,
    make_pairs,
    markdown,
    progress,
    require_complete,
)


def test_no_interim_analysis_even_if_a_result_file_exists(tmp_path):
    write_json(tmp_path / "status.json", {"stage": "final_50_52", "state": "running"})
    (tmp_path / "final_result.json").write_text("not even JSON: must not be read")
    engine = SimpleNamespace(verify=lambda: pytest.fail("No auditing before completion"))
    with pytest.raises(RuntimeError, match="not completed"):
        audit(tmp_path, engine, None)


@pytest.mark.parametrize("state", ["not_proven", "significant_uplift"])
def test_terminal_negative_results_are_valid_completions(tmp_path, state):
    write_json(tmp_path / "status.json", {"stage": "complete", "state": state})
    assert require_complete(tmp_path)["state"] == state


def test_progress_counts_latest_attempts_without_reading_outcomes(tmp_path):
    write_json(tmp_path / "status.json", {"stage": "final_50_52", "state": "running"})
    root = tmp_path / "evaluations/final/s50/null/gpu4"
    write_json(root / "job.json", {"condition": "null", "tasks": ["a"]})
    for attempt, count in ((1, 5), (2, 3)):
        for seed in range(count):
            path = root / f"rollouts/attempt-{attempt}/a/{seed}/manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("outcome content must not be read")
    result = progress(tmp_path)
    assert result["primary_validated"] == 0
    assert result["recorded_not_validated"] == 3
    write_json(root / "done.json", {"manifests": [None] * 20})
    result = progress(tmp_path)
    assert result["primary_validated"] == 20
    assert result["recorded_not_validated"] == 0
    assert "success" not in str(result)


@pytest.fixture
def cohort():
    plan = {"all_tasks": ["a", "b"], "final": {"seed_indices": [50, 51, 52], "episodes_per_index": 20}}
    rows = {role: {task: {100000 * (index + 1) + n: (Path("unused"), {"success": True})
                         for index in (50, 51, 52) for n in range(20)}
                   for task in (["a"] if role == "negative" else ["a", "b"])}
            for role in ("null", "positive", "negative")}
    return plan, rows


def test_exit_race_does_not_mislabel_completed_study(tmp_path, monkeypatch):
    import scripts.recap_finalize_study as module
    write_json(tmp_path / "status.json", {"stage": "final_50_52", "state": "running", "pid": 1})

    def finish_then_exit(pid, signal):
        write_json(tmp_path / "status.json", {"stage": "complete", "state": "not_proven", "pid": pid})
        raise ProcessLookupError

    monkeypatch.setattr(module.os, "kill", finish_then_exit)
    assert controller_alive_or_terminal(tmp_path)


def test_absent_controller_is_not_restarted(tmp_path, monkeypatch):
    import scripts.recap_finalize_study as module
    write_json(tmp_path / "status.json", {"stage": "final_50_52", "state": "running", "pid": 1})

    def gone(pid, signal):
        assert signal == 0
        raise ProcessLookupError

    monkeypatch.setattr(module.os, "kill", gone)
    assert not controller_alive_or_terminal(tmp_path)


def test_pairs_preserve_all_frozen_strata(cohort):
    plan, rows = cohort
    pairs = make_pairs(plan, rows, ["a"])
    assert len(pairs) == 2
    for task in pairs:
        assert len(pairs[task]) == 60
        assert {r["seed_index"] for r in pairs[task]} == {50, 51, 52}


@pytest.mark.parametrize("error", ["missing_task", "unpaired", "wrong_index", "unequal_strata", "missing_negative"])
def test_incomplete_or_changed_cohorts_cannot_be_reported(cohort, error):
    plan, rows = cohort
    if error == "missing_task":
        del rows["positive"]["b"]
    if error == "unpaired":
        rows["positive"]["a"][5100040] = rows["positive"]["a"].pop(5100000)
    if error == "wrong_index":
        for role in rows:
            rows[role]["a"][5400000] = rows[role]["a"].pop(5100000)
    if error == "unequal_strata":
        for role in rows:
            rows[role]["a"][5200040] = rows[role]["a"].pop(5100000)
    if error == "missing_negative":
        rows["negative"] = {}
    with pytest.raises(ValueError):
        make_pairs(plan, rows, ["a"])


def test_report_includes_all_task_contributions_and_correct_units():
    pairs = {f"task{i:02}": [{"seed_index": 50 + j // 20, "positive": True,
                              "null": not (i == 0 and j < 12)} for j in range(60)] for i in range(50)}
    report = final_statistics(pairs, tasks=list(pairs), registry_tasks=["task00"])
    report["secondary_negative"] = {"task00": {"positive_success": 30, "null_success": 48, "episodes": 60}}
    text = markdown(report, Path("study"))
    assert "macro change: +0.4000 percentage points" in text
    assert "| task00 | yes | 48/60 | 60/60 | +20.000 | +0.4000 |" in text
    assert text.count("| task") == 50
    assert "NOT improvement on every task" in text
    assert "Negative 30/60; Null 48/60" in text
    assert "No production registry promotion" in text
