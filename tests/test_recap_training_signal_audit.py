from types import SimpleNamespace

import numpy as np
import pytest

from scripts.recap_audit_training_signal import describe, observation_delta, rms


class Arrays(dict):
    @property
    def files(self):
        return list(self)


def test_audit_separates_exact_observations_from_tolerance_drift():
    a = Arrays({"observation::state": np.zeros(14, dtype=np.float32),
                "observation::images.cam": np.zeros((2, 2, 3), dtype=np.uint8)})
    b = Arrays({k: v.copy() for k, v in a.items()})
    assert observation_delta(a, b)["byte_identical"]
    b["observation::state"][0] = 0.0001
    b["observation::images.cam"][0, 0, 0] = 12
    d = observation_delta(a, b)
    assert not d["byte_identical"]
    assert d["numeric_max_abs"] == pytest.approx(0.0001)
    assert d["image_mae_max"] == 1
    assert rms(a["observation::state"], b["observation::state"]) > 0


def test_audit_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="incompatible"):
        observation_delta(Arrays({"observation::state": np.zeros(14)}),
                          Arrays({"observation::state": np.zeros(15)}))


def test_missing_or_nonfinite_arrays_fail_closed():
    good = Arrays({"observation::state": np.zeros(14)})
    with pytest.raises(ValueError, match="keys"):
        observation_delta(good, Arrays())
    with pytest.raises(ValueError, match="Nonfinite"):
        observation_delta(good, Arrays({"observation::state": np.full(14, np.nan)}))
    with pytest.raises(ValueError, match="compatible finite"):
        rms(np.zeros(14), np.full(14, np.nan))


def test_distribution_summary_does_not_call_rows_independent_states():
    assert describe([]) == {"count": 0}
    assert describe([50] * 13) == {"count": 13, "min": 50, "max": 50, "mean": 50, "median": 50}


def test_legacy_initializer_has_two_dominant_directions():
    """Legacy compatibility preserves this defect; new training opts into full rank."""
    import ast
    from pathlib import Path
    torch = pytest.importorskip("torch")
    path = Path(__file__).parents[1] / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlowMatchingV2")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "reset_recap_adapter")
    from lingbotvla.recap.initialization import initialize_velocity_lora_
    namespace = {"torch": torch, "initialize_velocity_lora_": initialize_velocity_lora_}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    a, b = torch.empty(2, 8, 768), torch.empty(2, 55, 8)
    obj = SimpleNamespace(recap_adapter_enabled=True, recap_adapter_type="velocity_lora",
                          recap_velocity_lora_a=a, recap_velocity_lora_b=b,
                          config=SimpleNamespace(recap_adapter_init_std=0.02))
    namespace["reset_recap_adapter"](obj)
    s = torch.linalg.svdvals(a[1].double())
    assert s[2] / s[0] < 1e-5
    assert (s[:2].square().sum() / s.square().sum()).item() > 0.99999999
    assert torch.count_nonzero(b) == 0
