import json
from types import SimpleNamespace

import pytest

from scripts import recap_wait_backend_probe as queue


def test_waits_on_resources_then_launches_exactly_once(tmp_path, monkeypatch):
    (tmp_path / "seal.json").write_text("{}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    states = iter(["waiting_for_frozen_study", "waiting_for_gpu", "ready"])
    monkeypatch.setattr(queue, "readiness", lambda *args: next(states))
    monkeypatch.setattr(queue.time, "sleep", lambda *args: None)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(queue.subprocess, "run", run)
    assert queue.wait_and_run(tmp_path, tmp_path, 4) == 0
    assert len(calls) == 1 and "--run" in calls[0] and "--gpu" in calls[0]
    assert json.loads((tmp_path / "queue_result.json").read_text())["retries"] == 0
    with pytest.raises(queue.ProbeWaitFailure, match="no restart"):
        queue.wait_and_run(tmp_path, tmp_path, 4)
    assert len(calls) == 1


def test_failed_probe_is_not_retried(tmp_path, monkeypatch):
    (tmp_path / "seal.json").write_text("{}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setattr(queue, "readiness", lambda *args: "ready")
    calls = []
    monkeypatch.setattr(
        queue.subprocess, "run", lambda *args, **kwargs: calls.append(args) or SimpleNamespace(returncode=78)
    )
    assert queue.wait_and_run(tmp_path, tmp_path, 4) == 78
    assert len(calls) == 1
    assert not json.loads((tmp_path / "queue_result.json").read_text())["training_launched"]


def test_wait_deadline_stops_without_launch(tmp_path, monkeypatch):
    (tmp_path / "seal.json").write_text("{}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setattr(queue, "readiness", lambda *args: "waiting_for_gpu")
    times = iter([0, 10])
    monkeypatch.setattr(queue.time, "monotonic", lambda: next(times))
    with pytest.raises(queue.ProbeWaitFailure, match="expired"):
        queue.wait_and_run(tmp_path, tmp_path, 4, max_wait_seconds=1)
    assert not (tmp_path / "launch_once.json").exists()


@pytest.mark.parametrize(
    "status",
    [
        {"stage": "stopped", "state": "infrastructure_or_protocol_failure"},
        {"stage": "final_50_52", "state": "infrastructure_or_protocol_failure_waiting_for_workers"},
    ],
)
def test_failed_study_requires_manual_audit(tmp_path, status):
    (tmp_path / "status.json").write_text(json.dumps(status))
    with pytest.raises(queue.ProbeWaitFailure, match="manual audit"):
        queue.readiness(tmp_path, 4)


@pytest.mark.parametrize("outcome", ["significant_uplift", "not_proven"])
def test_readiness_does_not_depend_on_final_outcome(tmp_path, monkeypatch, outcome):
    (tmp_path / "status.json").write_text(json.dumps({"stage": "complete", "state": outcome}))
    (tmp_path / "final_result.json").write_text("MUST NOT BE READ")
    monkeypatch.setattr(queue, "require_idle_gpu", lambda *args: None)
    assert queue.readiness(tmp_path, 4) == "ready"
