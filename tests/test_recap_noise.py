import pytest

from lingbotvla.recap.noise import policy_action_seed


def _seed(**overrides):
    values = {
        "policy_seed": 600,
        "continuation_policy_seed": 100,
        "counterfactual_decision": 1,
        "decision_index": 0,
        "task_name": "click_bell",
        "environment_seed": 123,
        "per_episode": True,
    }
    values.update(overrides)
    return policy_action_seed(**values)


def test_episode_common_noise_is_repeatable_and_condition_independent():
    assert _seed() == _seed()
    assert _seed(decision_index=1) == _seed(decision_index=1)


def test_episode_common_noise_changes_across_task_environment_and_decision():
    values = {
        _seed(),
        _seed(task_name="turn_switch"),
        _seed(environment_seed=124),
        _seed(decision_index=2),
        _seed(policy_seed=601, decision_index=1),
    }
    assert len(values) == 5


def test_legacy_schedule_is_preserved():
    assert _seed(per_episode=False, decision_index=1) == 600
    assert _seed(per_episode=False, decision_index=0) == 100
    assert _seed(per_episode=False, decision_index=2) == 102
    assert _seed(continuation_policy_seed=None) is None


def test_episode_schedule_requires_reset_identity():
    with pytest.raises(ValueError, match="canonical task"):
        _seed(task_name=None)
    with pytest.raises(ValueError, match="environment seed"):
        _seed(environment_seed=None)
