"""Fail-closed guards for deployment-equivalent, adapter-only FM training.

Only teacher-forced x_t independent of trainable parameters may be detached.
This is NOT a differentiable frozen backbone for end-to-end sampler training.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch


DEPLOYMENT_FEATURE_PROTOCOL = "recap-deployment-features-v1"
TRAINING_BACKENDS = ("legacy", "deployment")
ADAPTER_PARAMETERS = {"recap_velocity_lora_a", "recap_velocity_lora_b"}


def validate_deployment_config(config):
    if not (
        getattr(config, "train_recap_adapter_only", False)
        and getattr(config, "recap_adapter_enabled", False)
        and getattr(config, "recap_adapter_type", None) == "velocity_lora"
    ):
        raise ValueError("Deployment-feature training requires adapter-only velocity_lora")
    if getattr(config, "recap_prompt_enabled", True):
        raise ValueError("Deployment-feature training requires recap_prompt_enabled=False")
    if getattr(config, "use_compile", False):
        raise ValueError("Deployment-feature training requires eager execution")
    if getattr(config, "attention_implementation", None) != "eager":
        raise ValueError("Deployment features require the deployed eager attention implementation")
    if getattr(config, "enable_visual_distillation", True):
        raise ValueError("Disable visual distillation for deployment-feature training")
    if any(
        getattr(config, key, 0) != 0
        for key in ("sequence_wise_loss_coeff", "router_z_loss_coeff", "bias_update_speed")
    ):
        raise ValueError("Frozen deployment features require zero router losses and bias updates")
    if getattr(config, "loss_type", None) not in ("fm", "L1_fm"):
        raise ValueError("Deployment-feature training supports fm and L1_fm only")


def validate_deployment_training_arguments(args):
    """Validate before CUDA/model initialization, not after an expensive launch."""
    train, data = args.train, args.data
    backend = getattr(train, "recap_training_backend", "legacy")
    if backend not in TRAINING_BACKENDS:
        raise ValueError(f"Unknown recap_training_backend: {backend}")
    if backend == "legacy":
        return
    from types import SimpleNamespace

    validate_deployment_config(SimpleNamespace(**{**vars(train), "recap_prompt_enabled": data.recap_prompt_enabled}))
    if not data.recap_enabled or not data.recap_adapter_enabled:
        raise ValueError("Deployment-feature training requires explicit data RECAP condition ids")
    if getattr(data, "image_augment", False):
        raise ValueError("Disable image augmentation for the deployment-equivalence training gate")
    if getattr(train, "enable_resume", False):
        raise ValueError("Deployment-feature v1 requires a fresh adapter-training run")
    if getattr(train, "optimizer", "adamw") != "adamw":
        raise ValueError("Deployment-feature v1 requires the audited AdamW recipe")
    if any(
        getattr(train, key, 1) != 1
        for key in (
            "tensor_parallel_size",
            "expert_parallel_size",
            "pipeline_parallel_size",
            "context_parallel_size",
            "ulysses_parallel_size",
            "data_parallel_shard_size",
        )
    ):
        raise ValueError("Deployment-feature v1 requires all parallel dimensions to be one")
    if (
        train.world_size != 1
        or train.data_parallel_mode != "ddp"
        or train.init_device != "cuda"
        or train.micro_batch_size != 1
    ):
        raise ValueError("Deployment-feature v1 requires single-device ddp, init_device=cuda, micro_batch_size=1")
    if not (train.enable_fp32 and train.enable_mixed_precision and train.enable_full_determinism):
        raise ValueError("Deployment-feature training requires FP32 parameters and strict determinism")
    if train.enable_gradient_checkpointing or train.enable_activation_offload or train.enable_fsdp_offload:
        raise ValueError("Deployment-feature v1 does not support checkpointing or offloading")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise ValueError("Set NVIDIA_TF32_OVERRIDE=0 before CUDA initialization")


def validate_frozen_feature_model(model, inputs):
    validate_deployment_config(model.config)
    if not getattr(model.config, "use_cache", False):
        raise ValueError("Deployment features require the real prefix KV cache")
    if torch.is_inference_mode_enabled():
        raise ValueError("Use no_grad, not inference_mode, for tensors consumed by a trainable adapter")
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise ValueError("Deployment features require strict deterministic algorithms (not warn-only)")
    parameters = dict(model.named_parameters())
    trainable = {name for name, p in parameters.items() if p.requires_grad}
    if trainable != ADAPTER_PARAMETERS:
        raise ValueError(f"Only velocity A/B may be trainable, found {sorted(trainable)}")
    devices = {p.device for p in parameters.values()}
    if len(devices) != 1 or next(iter(devices)).type not in ("cuda", "cpu"):
        raise ValueError("Deployment-feature v1 requires one materialized device")
    device = next(iter(devices))
    for name, p in parameters.items():
        if hasattr(p, "placements") or p.dtype != torch.float32:
            raise ValueError(f"Deployment-feature v1 requires unsharded FP32 parameters: {name}")
    for value in inputs:
        if value is None:
            continue
        if not isinstance(value, torch.Tensor) or value.requires_grad:
            raise ValueError("Frozen feature inputs must be tensors independent of trainable parameters")
        if value.device != device or (value.is_floating_point() and value.dtype != torch.float32):
            raise ValueError("Deployment feature inputs must match the FP32 model device")
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite deployment feature input")
    try:
        autocast = torch.is_autocast_enabled(device.type)
    except TypeError:  # PyTorch < 2.4 has separate CPU/CUDA query APIs.
        autocast = torch.is_autocast_cpu_enabled() if device.type == "cpu" else torch.is_autocast_enabled()
    if autocast:
        raise ValueError("Autocast is forbidden for deployment-feature training")
    if getattr(model, "_use_compile_predict_velocity", False):
        raise ValueError("Compiled velocity is forbidden for deployment-feature training")
    for module in model.modules():
        if getattr(module, "gradient_checkpointing", False):
            raise ValueError("Checkpointing is unsupported for deployment features")
        # Inspect the actual MoE blocks, not only a top-level YAML field.
        if hasattr(module, "num_experts") and hasattr(module, "experts"):
            if getattr(module, "_moe_implementation", None) != "fused":
                raise ValueError("Deployment features require the deployed fused/robby MoE storage")
    if device.type == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
        if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
            raise ValueError("Deployment features require NVIDIA_TF32_OVERRIDE=0")
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise ValueError("Deployment features require disabled TF32 backend flags")
    return device


@contextmanager
def frozen_deployment_mode(model, *inputs):
    """Temporarily select inference kernels; restore every module flag on error."""
    validate_frozen_feature_model(model, inputs)
    modes = [(module, module.training) for module in model.modules()]
    blocks = [
        (name, module, getattr(module, "_recap_capture_backend", False))
        for name, module in model.named_modules()
        if hasattr(module, "num_experts") and hasattr(module, "experts")
    ]
    if getattr(model.config, "use_moe", False) and not blocks:
        raise ValueError("MoE configuration has no auditable routed blocks")
    try:
        for module, _ in modes:
            module.training = False
        for _, module, _ in blocks:
            module._recap_capture_backend = True
            module._recap_last_backend = None
        with torch.no_grad():
            yield
        telemetry = {}
        for name, module, _ in blocks:
            row = module._recap_last_backend
            if (
                row is None
                or row["backend"] != "robby_deterministic"
                or row["input_dtype"] != "torch.float32"
                or row["routed_output_dtype"] != "torch.float32"
                or row["expert_parameter_dtype"] != "torch.float32"
                or row["training"]
                or row["grad_enabled"]
            ):
                raise ValueError(f"Deployment feature backend violation at {name}: {row}")
            telemetry[name] = dict(row)
        model._recap_backend_telemetry = telemetry
    finally:
        # Do not call train(previous_root_flag): children may originally differ.
        for module, mode in modes:
            module.training = mode
        for _, module, capture in blocks:
            module._recap_capture_backend = capture


def install_adapter_optimizer_guard(optimizer, adapter_parameters):
    """Reject nonfinite parameters/gradients before an optimizer can corrupt A/B."""
    expected = {id(p) for p in adapter_parameters}
    if len(expected) != 2:
        raise ValueError("Deployment-feature optimizer must train exactly A and B")

    def check(_optimizer, _args, _kwargs):
        trainable = [p for group in _optimizer.param_groups for p in group["params"] if p.requires_grad]
        if len(trainable) != 2 or {id(p) for p in trainable} != expected:
            raise ValueError("Deployment-feature optimizer must train exactly A and B")
        for group in _optimizer.param_groups:
            for parameter in group["params"]:
                if not parameter.requires_grad:
                    if parameter.grad is not None:
                        raise ValueError("Frozen base acquired an optimizer gradient")
                    continue
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError("Nonfinite adapter parameter before optimizer step")
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Nonfinite adapter gradient before optimizer step")

    return optimizer.register_step_pre_hook(check)


def validate_feature_outputs(hidden, velocity):
    if hidden.requires_grad or velocity.requires_grad or hidden.is_inference() or velocity.is_inference():
        raise ValueError("Frozen features must be ordinary detached tensors")
    if hidden.dtype != torch.float32 or velocity.dtype != torch.float32:
        raise ValueError("Frozen features must remain FP32; refusing a precision fallback")
    if hidden.shape[:2] != velocity.shape[:2] or hidden.ndim != 3 or velocity.ndim != 3:
        raise ValueError("Frozen features must have matching [B,T,H/D] shapes")
    if not torch.isfinite(hidden).all() or not torch.isfinite(velocity).all():
        raise ValueError("Nonfinite frozen deployment features")
