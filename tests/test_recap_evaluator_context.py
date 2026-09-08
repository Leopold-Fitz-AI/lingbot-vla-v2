import numpy as np
import pytest

from deploy.recap_evaluator_context import (
    capture_evaluator_context,
    restore_evaluator_context,
    validate_evaluator_context,
)


class ArmTag(str):
    pass


class Environment:
    pass


@pytest.mark.parametrize("scalar", [np.float32(0.834917), np.float64(0.834917), 0.834917])
def test_evaluator_context_preserves_native_arm_and_scalar_type(scalar):
    expert = Environment()
    expert.arm_tag, expert.origin_z = ArmTag("right"), scalar
    saved = capture_evaluator_context(expert, "put_object_cabinet")
    fresh = Environment()
    restore_evaluator_context(fresh, "put_object_cabinet", saved)
    assert type(fresh.arm_tag) is ArmTag
    assert type(fresh.origin_z) is type(scalar)
    assert np.asarray(fresh.origin_z).tobytes() == np.asarray(scalar).tobytes()
    assert capture_evaluator_context(fresh, "put_object_cabinet") == saved


@pytest.mark.parametrize("task", ["open_laptop", "place_object_scale", "put_object_cabinet"])
def test_missing_context_is_not_silently_accepted(task):
    with pytest.raises(ValueError, match="context"):
        validate_evaluator_context(task, {})


def test_unknown_context_cannot_modify_an_evaluator_or_policy():
    with pytest.raises(ValueError, match="context"):
        restore_evaluator_context(Environment(), "bell", {"success_threshold": 0})
    with pytest.raises(ValueError, match="origin_z"):
        validate_evaluator_context("put_object_cabinet", {
            "arm_tag": "left", "origin_z": {"kind": "numpy_scalar", "dtype": "float32", "value": float("nan")}})
    assert capture_evaluator_context(Environment(), "bell") == {}
