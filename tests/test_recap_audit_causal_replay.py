import json

import numpy as np
import pytest

from scripts.recap_audit_causal_replay import audit_replay, build_protocol, protocol_maps


def _write(root, name, *, success, paired=True, text="collected instruction", continuation=200,
           decision=1, state_value=0.0):
    folder = root / name
    folder.mkdir(parents=True)
    np.savez(folder / "step.npz", **{
        "observation::observation.state": np.full(14, state_value, dtype=np.float32),
        "observation::observation.images.head": np.zeros((2, 2, 3), dtype=np.uint8),
    })
    metadata = {
        "task_name": "bell", "recap_condition": "null", "continuation_policy_seed": str(continuation),
        "counterfactual_policy_decision": decision, "common_noise_per_episode": "true",
        "policy_seed": "900",
    }
    if paired:
        metadata.update(paired_environment_seed=3300002, paired_original_decision_index=decision)
    row = {
        "task": text, "seed": 3300002, "success": success, "metadata": metadata,
        "steps": [{"decision_index": 0 if paired else 1, "file": "step.npz"}],
    }
    path = folder / "manifest.json"
    path.write_text(json.dumps(row))
    return path


def _protocol(tmp_path):
    root = tmp_path / "paired"
    _write(root, "positive", success=True)
    _write(root, "negative", success=False)
    return build_protocol([root], policy_seed=900)


def test_freezes_literal_text_and_collected_continuation_not_validation_defaults(tmp_path):
    protocol = _protocol(tmp_path)
    assert protocol["continuation_policy_seed"] == 200
    assert protocol["counterfactual_decisions"] == {"bell": 1}
    maps = protocol_maps(protocol)
    assert maps["instructions.json"]["tasks"]["bell"] == {"3300002": "collected instruction"}
    assert maps["environment_seeds.json"]["tasks"]["bell"] == [3300002]
    root = tmp_path / "replay"
    _write(root, "null", success=True, paired=False)
    report = audit_replay(protocol, root)
    assert report["passed"]
    assert report["matched_states"] == 1


def test_same_environment_seed_does_not_imply_same_causal_state(tmp_path):
    protocol = _protocol(tmp_path)
    root = tmp_path / "wrong_replay"
    _write(root, "null", success=True, paired=False, text="different text", continuation=300,
           decision=0, state_value=0.02)
    report = audit_replay(protocol, root)
    assert not report["passed"]
    assert report["mismatch_counts"] == {
        "instruction": 1, "continuation_policy_seed": 1,
        "counterfactual_policy_decision": 1, "intervention_observation": 1,
    }


def test_missing_states_and_latest_attempt_are_checked(tmp_path):
    protocol = _protocol(tmp_path)
    root = tmp_path / "replay"
    assert audit_replay(protocol, root)["mismatch_counts"] == {"missing_episode": 1}
    _write(root / "attempt-1", "null", success=True, paired=False, text="bad old text")
    _write(root / "attempt-2", "null", success=False, paired=False)
    assert audit_replay(protocol, root)["passed"]
    _write(root / "attempt-2", "duplicate", success=False, paired=False)
    with pytest.raises(ValueError, match="Duplicate"):
        audit_replay(protocol, root)


def test_rejects_conflicting_instructions_or_continuations(tmp_path):
    root = tmp_path / "paired"
    _write(root, "positive", success=True)
    path = _write(root, "negative", success=False, text="bad")
    with pytest.raises(ValueError, match="Conflicting collected instructions"):
        build_protocol([root], policy_seed=900)
    row = json.loads(path.read_text())
    row["task"] = "collected instruction"
    row["metadata"]["continuation_policy_seed"] = "300"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="continuation schedules"):
        build_protocol([root], policy_seed=900)


def test_non_mixed_source_and_early_terminal_fail_explicitly(tmp_path):
    root = tmp_path / "positive_only"
    _write(root, "positive", success=True)
    with pytest.raises(ValueError, match="Not a mixed-outcome"):
        build_protocol([root], policy_seed=900)
    protocol = _protocol(tmp_path)
    replay = tmp_path / "replay"
    path = _write(replay, "null", success=True, paired=False)
    row = json.loads(path.read_text())
    row["steps"][0]["decision_index"] = 0
    path.write_text(json.dumps(row))
    assert audit_replay(protocol, replay)["mismatch_counts"] == {"intervention_not_reached": 1}
