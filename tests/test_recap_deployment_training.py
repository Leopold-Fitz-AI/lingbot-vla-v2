"""Production methods with a tiny backbone: no 6B allocation or CUDA import.

GPU full-model parity remains a separate mandatory gate, not implied by mocks.
"""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from lingbotvla.recap.adapter import apply_recap_velocity_lora
from lingbotvla.recap.initialization import initialize_velocity_lora_
from lingbotvla.recap.training import (
    DEPLOYMENT_FEATURE_PROTOCOL,
    TRAINING_BACKENDS,
    frozen_deployment_mode,
    validate_deployment_training_arguments,
    validate_feature_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"


def production_class():
    tree = ast.parse(MODEL.read_text())
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlowMatchingV2")


def test_refactor_preserves_pre_patch_inference_arithmetic():
    # AST hashes from commit 454efc9. Exclude only the velocity docstring and
    # final adapter call (now in the wrapper). Expand the extracted prefix helper.
    methods = {n.name: n for n in production_class().body if isinstance(n, ast.FunctionDef)}

    def digest(body):
        return hashlib.sha256(ast.dump(ast.Module(body=body, type_ignores=[])).encode()).hexdigest()

    assert (
        digest(methods["predict_velocity_features"].body[1:-1])
        == "c38da4977b87040073b7dc10a1ae3160d4ed1b862b84caad0938983534ba20da"
    )
    body = []
    for n in methods["sample_actions"].body:
        if (
            isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "prepare_velocity_prefix"
        ):
            body.extend(methods["prepare_velocity_prefix"].body[1:-1])
        else:
            body.append(n)
    assert digest(body) == "23824c80b69f367269129d97cc59e2e4c50ee74ab9684abcaea2f67f894f5d57"


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.calls = []

    def forward(self, *, inputs_embeds, past_key_values, **kwargs):
        self.calls.append((self.training, torch.is_grad_enabled(), inputs_embeds[0] is None))
        prefix, suffix = inputs_embeds
        if prefix is not None:
            cache = self.linear(prefix).mean(1, keepdim=True)
        else:
            cache = past_key_values
        result = None if suffix is None else self.linear(suffix) + cache
        return (prefix, result), cache, []


class TinyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            train_recap_adapter_only=True,
            recap_adapter_enabled=True,
            recap_adapter_type="velocity_lora",
            recap_adapter_scale=2,
            recap_signed_velocity_axis=True,
            recap_adapter_initialization="orthogonal_matched_v1",
            recap_adapter_init_seed=971,
            recap_prompt_enabled=False,
            use_compile=False,
            enable_visual_distillation=False,
            recap_training_backend="deployment",
            sequence_wise_loss_coeff=0,
            router_z_loss_coeff=0,
            bias_update_speed=0,
            loss_type="L1_fm",
            n_action_steps=3,
            max_action_dim=3,
            use_cache=True,
            attention_implementation="eager",
            adanorm_time=False,
            align_params={},
            num_steps=2,
        )
        self.recap_adapter_enabled, self.recap_adapter_type = True, "velocity_lora"
        self.qwenvl_with_expert = TinyBackbone()
        self.state_proj, self.action_in_proj, self.action_out_proj = nn.Linear(3, 4), nn.Linear(3, 4), nn.Linear(4, 3)
        self.recap_velocity_lora_a = nn.Parameter(torch.empty(2, 2, 4))
        self.recap_velocity_lora_b = nn.Parameter(torch.empty(2, 3, 2))
        self.reset_recap_adapter()
        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith("recap_velocity_lora_"))
        self.block_future_depth_to_action = False

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, image_grid_thw=None):
        batch = images.shape[0]
        prefix = images.reshape(batch, -1, 4)
        mask = torch.ones(prefix.shape[:2], dtype=torch.bool)
        positions = torch.arange(prefix.shape[1]).view(1, 1, -1).expand(3, batch, -1)
        return prefix, mask, torch.zeros_like(mask), positions, mask, []

    def embed_suffix(self, state, x_t, timestep, recap_condition_id=None):
        suffix = torch.cat((self.state_proj(state)[:, None], self.action_in_proj(x_t) + timestep[:, None, None]), 1)
        mask = torch.ones(suffix.shape[:2], dtype=torch.bool)
        att = torch.zeros_like(mask)
        att[:, :2] = True
        return timestep, suffix, mask, att

    def _block_suffix_to_future_video_if_enabled_(self, masks, **kwargs):
        return masks

    def _moe_losses_and_metrics(self, routers, loss):
        return loss.new_zeros(()), loss.new_zeros(()), {}


def install_production_methods():
    utils = ast.parse((ROOT / "lingbotvla/models/vla/lingbot_vla/utils.py").read_text())
    mask = next(n for n in utils.body if isinstance(n, ast.FunctionDef) and n.name == "make_att_2d_masks")
    names = {
        "reset_recap_adapter",
        "_apply_recap_velocity_adapter",
        "_build_full_position_ids",
        "recap_frozen_features",
        "prepare_velocity_prefix",
        "predict_velocity_features",
        "predict_velocity",
        "forward",
        "sample_actions",
    }
    scope = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "F": F,
        "apply_recap_velocity_lora": apply_recap_velocity_lora,
        "initialize_velocity_lora_": initialize_velocity_lora_,
        "TRAINING_BACKENDS": TRAINING_BACKENDS,
        "DEPLOYMENT_FEATURE_PROTOCOL": DEPLOYMENT_FEATURE_PROTOCOL,
        "frozen_deployment_mode": frozen_deployment_mode,
        "validate_feature_outputs": validate_feature_outputs,
    }
    methods = [n for n in production_class().body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=[mask, *methods], type_ignores=[]), str(MODEL), "exec"), scope)
    for n in methods:
        setattr(TinyFlow, n.name, scope[n.name])


install_production_methods()


@pytest.fixture
def model():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield TinyFlow()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(22)
    return dict(
        images=torch.randn(1, 2, 4, generator=g),
        img_masks=torch.ones(1, 2, dtype=torch.bool),
        lang_tokens=torch.ones(1, 2, dtype=torch.long),
        lang_masks=torch.ones(1, 2, dtype=torch.bool),
        state=torch.randn(1, 3, generator=g),
        actions=torch.randn(1, 3, 3, generator=g),
        noise=torch.randn(1, 3, 3, generator=g),
        time=torch.tensor([0.5]),
        recap_condition_id=torch.tensor([1]),
    )


def feature_inputs(batch):
    result = {k: v for k, v in batch.items() if k not in ("actions", "noise", "time", "recap_condition_id")}
    t = batch["time"][:, None, None]
    return {**result, "x_t": t * batch["noise"] + (1 - t) * batch["actions"], "timestep": batch["time"]}


def test_real_shared_methods_match_inference_and_restore_modes(model, batch):
    model.train()
    model.qwenvl_with_expert.linear.eval()
    before = [m.training for m in model.modules()]
    h, v = model.recap_frozen_features(**feature_inputs(batch))
    assert [m.training for m in model.modules()] == before
    assert all(not train and not grad for train, grad, _ in model.qwenvl_with_expert.calls)
    assert not h.requires_grad and not v.requires_grad
    model.eval()
    with torch.no_grad():
        pad, pos, cache = model.prepare_velocity_prefix(
            batch["images"], batch["img_masks"], batch["lang_tokens"], batch["lang_masks"]
        )
        expected = model.predict_velocity(
            batch["state"],
            pad,
            cache,
            feature_inputs(batch)["x_t"],
            batch["time"],
            prefix_position_ids=pos,
            recap_condition_id=[1],
        )
    assert torch.equal(v, expected)
    assert model._recap_feature_protocol == DEPLOYMENT_FEATURE_PROTOCOL


def test_training_reaches_b_then_a_without_updating_base(model, batch):
    base = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("recap_")}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.2)
    for step in range(2):
        optimizer.zero_grad()
        losses = model(**batch, loss_type="L1_fm")[0]
        assert losses.requires_grad
        losses.mean().backward()
        assert model.recap_velocity_lora_b.grad.abs().max() > 0
        if step == 1:
            assert model.recap_velocity_lora_a.grad.abs().max() > 0
        assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith("recap_"))
        optimizer.step()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in base.items())
    batch["recap_condition_id"] = torch.tensor([-1])
    losses = model(**batch)[0]
    assert torch.count_nonzero(model._last_recap_velocity_residual) == 0
    losses.mean().backward()


def test_policy_wrapper_preserves_masked_bc_and_regularization(model, batch):
    from lingbotvla.recap.loss import masked_action_loss

    cls = next(
        n for n in ast.parse(MODEL.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "LingbotVlaV2Policy"
    )
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    scope = {"torch": torch, "Tensor": torch.Tensor, "masked_action_loss": masked_action_loss}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(MODEL), "exec"), scope)

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.config = model, model.config

    Wrapper.forward = scope["forward"]
    model.config.action_dim = 3
    model.config.recap_residual_loss_weight = 100
    with torch.no_grad():
        model.recap_velocity_lora_b.fill_(0.1)
    wrapper = Wrapper()
    result = wrapper(**batch, action_is_pad=torch.tensor([[False, True, True]]))
    hidden, base = model.recap_frozen_features(**feature_inputs(batch))
    velocity = model._apply_recap_velocity_adapter(hidden, base, [1])
    residual = model._last_recap_velocity_residual
    expected = (batch["noise"] - batch["actions"] - velocity)[:, 0].abs().mean() + 100 * residual[:, 0].square().mean()
    assert torch.equal(result[0], expected)
    result[0].backward()
    assert model.recap_velocity_lora_a.grad.abs().max() > 0


def test_legacy_path_does_not_select_deployment_mode(model, batch):
    model.config.recap_training_backend = "legacy"
    model(**batch)
    assert any(train and grad for train, grad, _ in model.qwenvl_with_expert.calls)


@pytest.mark.parametrize(
    "field,value,pattern",
    [
        ("recap_prompt_enabled", True, "prompt"),
        ("train_recap_adapter_only", False, "adapter-only"),
        ("recap_adapter_type", "embedding", "velocity_lora"),
        ("use_compile", True, "eager"),
        ("attention_implementation", "flex_cached", "eager attention"),
        ("use_cache", False, "KV cache"),
        ("bias_update_speed", 0.001, "bias"),
        ("enable_visual_distillation", True, "distillation"),
        ("loss_type", "repeat_fm", "L1_fm"),
        ("recap_training_backend", "typo", "Unknown"),
    ],
)
def test_unsupported_configuration_fails_closed(model, batch, field, value, pattern):
    setattr(model.config, field, value)
    with pytest.raises(ValueError, match=pattern):
        model(**batch)
    assert not model.qwenvl_with_expert.calls


def test_grad_dependent_input_precision_or_extra_trainable_base_rejected(model, batch):
    batch["actions"].requires_grad_(True)
    with pytest.raises(ValueError, match="independent"):
        model(**batch)
    batch["actions"].requires_grad_(False)
    model.state_proj.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="Only velocity"):
        model(**batch)
    model.state_proj.weight.requires_grad_(False)
    model.state_proj.bfloat16()
    with pytest.raises(ValueError, match="FP32 parameters"):
        model(**batch)


def test_autocast_inference_mode_warn_only_and_nan_rejected(model, batch):
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="Autocast"):
        model(**batch)
    with torch.inference_mode(), pytest.raises(ValueError, match="inference_mode"):
        model(**batch)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.raises(ValueError, match="warn-only"):
        model(**batch)
    torch.use_deterministic_algorithms(True, warn_only=False)
    batch["images"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        model(**batch)


def test_flags_restore_after_backbone_error_and_aux_targets_not_ignored(model, batch, monkeypatch):
    before = [m.training for m in model.modules()]

    def fail(**kwargs):
        raise RuntimeError("injected backbone error")

    monkeypatch.setattr(model.qwenvl_with_expert, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        model(**batch)
    assert [m.training for m in model.modules()] == before
    with pytest.raises(ValueError, match="auxiliary"):
        model(**batch, depth_targets=torch.ones(1))


def test_features_own_storage_and_all_outputs_are_finite_fp32(model, batch):
    h, v = model.recap_frozen_features(**feature_inputs(batch))
    h.fill_(999)
    h2, v2 = model.recap_frozen_features(**feature_inputs(batch))
    assert not torch.equal(h, h2) and torch.equal(v, v2)
    with pytest.raises(ValueError, match="precision"):
        validate_feature_outputs(h2.bfloat16(), v2)
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_feature_outputs(h2, torch.full_like(v2, float("inf")))


def test_missing_actual_moe_telemetry_is_not_silently_accepted(model, batch):
    block = nn.Module()
    block.num_experts = 2
    block._moe_implementation = "fused"
    block.experts = nn.Linear(4, 4).requires_grad_(False)
    model.unexecuted_moe = block
    before = [m.training for m in model.modules()]
    with pytest.raises(ValueError, match="backend violation"):
        model(**batch)
    assert [m.training for m in model.modules()] == before
    assert not block._recap_capture_backend


def test_optimizer_rejects_nonfinite_gradient_before_mutating_parameters(model):
    from lingbotvla.recap.training import install_adapter_optimizer_guard

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.001)
    install_adapter_optimizer_guard(optimizer, parameters)
    before = [p.detach().clone() for p in parameters]
    parameters[0].grad = torch.full_like(parameters[0], float("nan"))
    with pytest.raises(FloatingPointError, match="gradient"):
        optimizer.step()
    assert all(torch.equal(p, b) for p, b in zip(parameters, before))
    assert not optimizer.state


def test_nonfinite_adapter_output_stops_training(model, batch):
    with torch.no_grad():
        model.recap_velocity_lora_b.fill_(float("inf"))
    with pytest.raises(FloatingPointError, match="target/velocity"):
        model(**batch)


def test_training_cli_checks_before_cuda(model, monkeypatch):
    train = SimpleNamespace(
        **{
            **vars(model.config),
            "world_size": 1,
            "data_parallel_mode": "ddp",
            "init_device": "cuda",
            "micro_batch_size": 1,
            "enable_fp32": True,
            "enable_mixed_precision": True,
            "enable_full_determinism": True,
            "enable_gradient_checkpointing": False,
            "enable_activation_offload": False,
            "enable_fsdp_offload": False,
        }
    )
    data = SimpleNamespace(recap_enabled=True, recap_adapter_enabled=True, recap_prompt_enabled=False)
    args = SimpleNamespace(train=train, data=data)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    validate_deployment_training_arguments(args)
    train.data_parallel_mode = "fsdp2"
    with pytest.raises(ValueError, match="single-device"):
        validate_deployment_training_arguments(args)
