"""Exercise production initialization without allocating the frozen 6B backbone."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from lingbotvla.recap.adapter import apply_recap_velocity_lora


ROOT = Path(__file__).resolve().parents[1]


def small_adapter(adapter_type):
    path = ROOT / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlowMatchingV2")
    constructor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    def assignment_to(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == name for t in node.targets)
    begin = next(i for i, n in enumerate(constructor.body) if assignment_to(n, "recap_adapter_enabled"))
    end = next(i for i, n in enumerate(constructor.body) if assignment_to(n, "align_params"))
    init = ast.parse("def initialize(self): pass").body[0]
    init.body = constructor.body[begin:end]
    reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "reset_recap_adapter")
    scope = {"torch": torch, "nn": nn}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[init, reset], type_ignores=[])), str(path), "exec"), scope)
    obj = nn.Module()
    obj.config = SimpleNamespace(recap_adapter_enabled=True, recap_adapter_type=adapter_type,
                                 recap_adapter_rank=2, proj_width=4, max_action_dim=3)
    obj.reset_recap_adapter = MethodType(scope["reset_recap_adapter"], obj)
    scope["initialize"](obj)
    return obj


def test_base_overlay_initializes_empty_memory_and_preserves_rng(monkeypatch):
    original = torch.empty
    def poisoned_empty(*args, **kwargs):
        return original(*args, **kwargs).fill_(float("nan"))
    monkeypatch.setattr(torch, "empty", poisoned_empty)
    rng_before = torch.get_rng_state().clone()
    obj = small_adapter("velocity_lora")
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.isfinite(obj.recap_velocity_lora_a).all()
    assert obj.recap_velocity_lora_a.abs().max() <= 0.02
    assert not obj.recap_velocity_lora_b.any()
    null = apply_recap_velocity_lora(torch.ones(1, 2, 4), [-1], obj.recap_velocity_lora_a,
                                     obj.recap_velocity_lora_b, signed_axis=True)
    assert torch.isfinite(null).all() and not null.any()


def test_loading_an_existing_adapter_overwrites_initialization():
    obj = small_adapter("velocity_lora")
    obj.load_state_dict({"recap_velocity_lora_a": torch.ones(2, 2, 4),
                         "recap_velocity_lora_b": torch.full((2, 3, 2), 0.1)})
    assert torch.equal(obj.recap_velocity_lora_a, torch.ones(2, 2, 4))
    assert torch.equal(obj.recap_velocity_lora_b, torch.full((2, 3, 2), 0.1))


def test_embedding_initialization_stays_zero():
    obj = small_adapter("embedding")
    assert not obj.recap_condition_embeddings.any()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_adapter_artifact_is_rejected(bad):
    path = ROOT / "deploy/lingbot_vla_v2_policy.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "LingbotVLAv2Server")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_recap_adapter_weights")
    class FakeFile:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def metadata(self):
            return {}
        def keys(self):
            return ["model.recap_velocity_lora_a"]
        def get_tensor(self, key):
            return torch.tensor([bad])
    scope = {"torch": torch, "Path": Path, "safe_open": lambda *args, **kwargs: FakeFile()}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    with pytest.raises(ValueError, match="Non-finite RECAP tensor"):
        scope[method.name](SimpleNamespace(), "unused.safetensors")
