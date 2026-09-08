import copy
from pathlib import Path

import numpy as np
import pytest

from lingbotvla.recap.evaluation import (
    exact_p,
    final_statistics,
    holm_adjust,
    initial_drift,
    validate_runtime,
    write_json,
)
from lingbotvla.recap.noise import policy_action_seed


def test_exact_p_and_holm_keep_failed_hypotheses():
    assert exact_p(0, 0) == 1
    assert exact_p(3, 0) == 0.25
    assert exact_p(10, 0) == 2 / 1024
    assert holm_adjust({"a": 0.01, "b": 0.02, "c": 0.2, "d": 1}) == {
        "a": 0.04, "b": 0.06, "c": 0.4, "d": 1}


def make_pairs(wins=0, guard_losses=0):
    return {f"task{i}": [{"seed_index": 50 + j // 20,
                          "positive": not (i == 1 and j < guard_losses),
                          "null": not (i == 0 and j < wins)} for j in range(60)] for i in range(50)}


def test_fixed_suite_claim_not_every_task_claim():
    pairs = make_pairs(wins=12)
    report = final_statistics(pairs, tasks=list(pairs), registry_tasks=["task0"])
    assert report["passed"]
    assert report["episodes_per_condition"] == 3000
    assert report["macro_delta"] == pytest.approx(0.004)  # NOT a +20 pp suite uplift!
    assert report["paired_bootstrap_95ci"][0] > 0
    assert len(report["tasks"]) == 50
    assert sum(r["delta"] > 0 for r in report["tasks"].values()) == 1


def test_identical_policies_are_not_significant():
    pairs = make_pairs()
    report = final_statistics(pairs, tasks=list(pairs), registry_tasks=[])
    assert not report["passed"]
    assert report["paired_bootstrap_95ci"] == [0, 0]
    assert report["two_sided_exact_p"] == 1


def test_material_guard_regression_blocks_success_even_if_aggregate_significant():
    pairs = make_pairs(wins=50, guard_losses=10)
    report = final_statistics(pairs, tasks=list(pairs), registry_tasks=["task0"])
    assert report["statistical_uplift"]
    assert not report["unadapted_guard_passed"]
    assert not report["passed"]


def test_missing_or_unequal_cohorts_are_errors():
    pairs = make_pairs()
    tasks = list(pairs)
    pairs["task0"].pop()
    with pytest.raises(ValueError, match="Incomplete/unequal"):
        final_statistics(pairs, tasks=tasks, registry_tasks=[])
    del pairs["task0"]
    with pytest.raises(ValueError, match="complete frozen"):
        final_statistics(pairs, tasks=tasks, registry_tasks=[])


def runtime_fixture():
    protocol = {"policy_seed": 900, "continuation_policy_seed": 300, "counterfactual_policy_decision": 0}
    runtime = {"schema_version": 1, "task_name": "bell", "environment_seed": 100,
               "decision_index": 0, "effective_condition": "null", "adapter_sha256": None,
               "use_compile": False, "use_bf16": False, "cfg_scale": 1.0,
               "condition_start_decision": 0, "condition_decisions": -1,
               "action_noise_seed": policy_action_seed(policy_seed=900, continuation_policy_seed=300,
                   counterfactual_decision=0, decision_index=0, task_name="bell", environment_seed=100, per_episode=True)}
    row = {"task": "press", "seed": 100, "metadata": {"task_name": "bell"}, "success": True,
           "steps": [{"decision_index": 0, "terminated": True, "executed_action_length": 25,
                      "generated_action_length": 50, "info": {"recap_runtime": runtime}}]}
    kwargs = dict(task="bell", seed=100, instruction="press", condition="positive",
                  registry={"tasks": {}}, protocol=protocol)
    return row, kwargs


def test_actual_server_null_fallback_not_client_requested_positive():
    row, kwargs = runtime_fixture()
    validate_runtime(row, **kwargs)
    row["steps"][0]["info"]["recap_runtime"]["effective_condition"] = "positive"
    with pytest.raises(ValueError, match="runtime mismatch"):
        validate_runtime(row, **kwargs)


@pytest.mark.parametrize("key,value", [("action_noise_seed", 901), ("use_compile", True),
                                      ("adapter_sha256", "wrong"), ("environment_seed", 101)])
def test_runtime_drift_rejected(key, value):
    row, kwargs = runtime_fixture()
    row["steps"][0]["info"]["recap_runtime"][key] = value
    with pytest.raises(ValueError, match="runtime mismatch"):
        validate_runtime(row, **kwargs)


def test_server_telemetry_is_required():
    row, kwargs = runtime_fixture()
    row["steps"][0]["info"] = {}
    with pytest.raises(ValueError, match="server-reported"):
        validate_runtime(row, **kwargs)


def test_image_drift_uses_mae_and_state_drift_uses_max(tmp_path):
    arrays = {"observation::observation.state": np.zeros(14, dtype=np.float32),
              "observation::observation.images.head": np.zeros((2, 2, 3), dtype=np.uint8)}
    np.savez(tmp_path / "a.npz", **arrays)
    changed = copy.deepcopy(arrays)
    changed["observation::observation.state"][0] = 0.0001
    changed["observation::observation.images.head"][0, 0, 0] = 12
    np.savez(tmp_path / "b.npz", **changed)
    drift = initial_drift((tmp_path / "a.json", {"steps": [{"file": "a.npz"}]}),
                          (tmp_path / "b.json", {"steps": [{"file": "b.npz"}]}))
    assert drift["numeric_max_abs"] == pytest.approx(0.0001)
    assert drift["image_mae_max"] == 1


def test_frozen_decisions_cannot_be_overwritten(tmp_path):
    path = Path(tmp_path) / "registry.json"
    write_json(path, {"tasks": {}}, immutable=True)
    write_json(path, {"tasks": {}}, immutable=True)
    with pytest.raises(ValueError, match="Frozen document changed"):
        write_json(path, {"tasks": {"bell": 1}}, immutable=True)


def test_strict_study_requires_actual_strict_runtime():
    row, kwargs = runtime_fixture()
    kwargs["protocol"]["deterministic_algorithms"] = True
    runtime = row["steps"][0]["info"]["recap_runtime"]
    with pytest.raises(ValueError, match="runtime mismatch"):
        validate_runtime(row, **kwargs)
    runtime.update(deterministic_algorithms=True, deterministic_warn_only=False)
    validate_runtime(row, **kwargs)
    runtime["deterministic_warn_only"] = True
    with pytest.raises(ValueError, match="runtime mismatch"):
        validate_runtime(row, **kwargs)


def test_locked_execution_requires_actual_cohort_evidence():
    from types import SimpleNamespace
    row, kwargs = runtime_fixture()
    kwargs["protocol"]["eligibility_protocol"] = "locked-expert-preflight-v1"
    hashes = {"state": "a" * 64}
    cohort = SimpleNamespace(sha256="f" * 64, entries={kwargs["seed"]: {"initial_observation_sha256": hashes}})
    kwargs["cohort"] = cohort
    with pytest.raises(ValueError, match="eligibility/runtime mismatch"):
        validate_runtime(row, **kwargs)
    row["metadata"].update(eligibility_protocol="locked-expert-preflight-v1", preflight_sha256=cohort.sha256,
                           initial_observation_sha256=hashes, evaluator_context={}, expert_rollouts_before_policy=0)
    validate_runtime(row, **kwargs)
    row["metadata"]["expert_rollouts_before_policy"] = 1
    with pytest.raises(ValueError, match="eligibility/runtime mismatch"):
        validate_runtime(row, **kwargs)
