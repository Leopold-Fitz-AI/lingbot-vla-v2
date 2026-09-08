"""Test the production response assembly without importing the multi-GB model."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_telemetry_does_not_mutate_cached_action_dictionary():
    path = Path(__file__).resolve().parents[1] / "deploy/lingbot_vla_v2_policy.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LingbotVLAv2Server")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "infer")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    action = np.arange(100, dtype=np.float32).reshape(1, 50, 2)
    outputs = {"action": action.copy()}
    policy = SimpleNamespace(chunk_ret=True, use_length=50, global_step=0,
                             last_action_chunk=None, last_normalized_action_chunk=None,
                             last_recap_runtime={"effective_condition": "null"},
                             _infer_batch=lambda observations: outputs)
    result = namespace["infer"](policy, {"batch": [{}]})
    assert set(result) == {"action", "_recap_runtime"}
    assert set(outputs) == {"action"}
    np.testing.assert_array_equal(result["action"], action)
    result["_recap_runtime"]["effective_condition"] = "tampered"
    assert policy.last_recap_runtime["effective_condition"] == "null"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_actions_never_reach_the_simulator(bad):
    path = Path(__file__).resolve().parents[1] / "deploy/lingbot_vla_v2_policy.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "LingbotVLAv2Server")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_unapply_batched_actions")
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    transform = SimpleNamespace(unapply=lambda _: {"action": torch.full((2, 2), bad)})
    policy = SimpleNamespace(action_key=["action"], use_bf16=False, vla=SimpleNamespace(feature_transform=transform))
    with pytest.raises(FloatingPointError, match="refusing to execute"):
        namespace[method.name](policy, [{}], torch.zeros(1, 2, 2))
