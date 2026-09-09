#!/usr/bin/env python3
"""Freeze consumed training/holdout inputs, then audit deployment feature parity.

No simulator, policy.sample_actions, optimizer, final outcomes, retries or
registry promotion. --prepare is CPU-only; --run refuses occupied GPUs and a
nonterminal frozen study. It must run from its separately frozen source tree.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from lingbotvla.recap.evaluation import sha256, write_json


TASK_STATES = {"hanging_mug": 13, "place_can_basket": 20, "stack_bowls_three": 17}
ADAPTERS = {"hanging_mug": "signed", "place_can_basket": "regularized", "stack_bowls_three": "positive"}
SOURCE_FILES = (
    "scripts/recap_audit_backend_parity.py",
    "deploy/lingbot_vla_v2_policy.py",
    "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py",
    "lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py",
    "lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py",
    "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py",
    "lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py",
    "lingbotvla/models/vla/lingbot_vla/utils.py",
    "lingbotvla/ops/robby_moe.py",
    "lingbotvla/ops/fused_moe.py",
    "lingbotvla/recap/adapter.py",
    "lingbotvla/recap/initialization.py",
    "lingbotvla/recap/training.py",
    "lingbotvla/data/vla_data/utils.py",
    "configs/robot_configs/robotwin.yaml",
)


def ensure_consumed_input(path, seed):
    if "final" in Path(path).parts or int(seed) // 100000 - 1 in (50, 51, 52):
        raise ValueError("Final evaluation inputs are forbidden in training diagnostics")


def validate_raw_chunk(path):
    with np.load(path, allow_pickle=False) as arrays:
        generated, executed = arrays["generated_action"], arrays["executed_action"]
        if (
            generated.shape != (50, 14)
            or not 0 < len(executed) <= 50
            or not np.isfinite(generated).all()
            or not np.array_equal(generated[: len(executed)], executed)
        ):
            raise ValueError("Invalid recorded generated/executed action chunk")
        observations = [k for k in arrays.files if k.startswith("observation::")]
        if not observations or any(not np.isfinite(arrays[k]).all() for k in observations):
            raise ValueError("Missing or nonfinite recorded observations")


def prepare(root, output, wrapper):
    """First sorted manifest per state, never filtered by success."""
    if output.exists():
        raise FileExistsError("Use a new diagnostic directory; never overwrite a frozen input lock")
    evidence, cases, adapters = {}, [], {}
    for task, expected_count in TASK_STATES.items():
        split_root = root / "state_splits" / task
        summary_path = split_root / "split_summary.json"
        summary = json.loads(summary_path.read_text())
        evidence[str(summary_path)] = sha256(summary_path)
        train_states, holdout_states = set(summary["train_states"]), set(summary["holdout_states"])
        if train_states & holdout_states or len(train_states | holdout_states) != expected_count:
            raise ValueError(f"Changed/incomplete consumed state split: {task}")
        selected = {}
        for split, expected in (("train", train_states), ("holdout", holdout_states)):
            observed = set()
            for path in sorted((split_root / split).rglob("manifest.json")):
                m = json.loads(path.read_text())
                seed = int(m["metadata"]["paired_environment_seed"])
                ensure_consumed_input(path, seed)
                if seed not in expected or m["metadata"]["task_name"] != task:
                    raise ValueError("State/task outside its original split")
                observed.add(seed)
                if seed in selected:
                    continue
                npz = path.parent / m["steps"][0]["file"]
                ensure_consumed_input(npz, seed)
                validate_raw_chunk(npz)
                evidence[str(path)], evidence[str(npz)] = sha256(path), sha256(npz)
                selected[seed] = {
                    "task": task,
                    "seed": seed,
                    "split": split,
                    "manifest": str(path),
                    "npz": str(npz),
                    "instruction": m["task"],
                    "decision": m["metadata"]["paired_original_decision_index"],
                    "pair_id": m["metadata"]["paired_match_id"],
                    "continuation_seed": m["metadata"]["continuation_policy_seed"],
                }
            if observed != expected:
                raise ValueError(f"Incomplete {task}/{split}; no shorter cohort permitted")
        cases.extend(selected[s] for s in sorted(selected))
        folder = root / "training_v2" / task / ADAPTERS[task]
        config_path = folder / "train_config.yaml"
        artifact = folder / "adapter/recap_adapter.safetensors"
        evidence[str(config_path)], evidence[str(artifact)] = sha256(config_path), sha256(artifact)
        adapters[task] = {"path": str(artifact), "config": str(config_path)}
    wrapper = wrapper.absolute()  # Preserve the wrapper's YAML ancestry across symlinks.
    # Hash dereferenced official weights plus the exact wrapper/preprocessor.
    for path in sorted(wrapper.glob("*.safetensors")):
        evidence[str(path)] = sha256(path)
    if not any(p.endswith(".safetensors") and str(wrapper) in p for p in evidence):
        raise ValueError("Wrapper contains no checkpoint shards")
    cli = wrapper.parents[2] / "lingbotvla_cli.yaml"
    evidence[str(cli)] = sha256(cli)
    import yaml

    wrapper_config = yaml.safe_load(cli.read_text())
    source = Path(__file__).resolve().parents[1]
    norm_value = wrapper_config["data"].get("norm_stats_file")
    if norm_value is None:
        # Exactly the server/FeatureTransform fallback, not an assumed override.
        robot_config = source / "configs/robot_configs/robotwin.yaml"
        norm_value = yaml.safe_load(robot_config.read_text())["norm_stats"]
    norm = Path(norm_value)
    norm = (source / norm).resolve() if not norm.is_absolute() else norm.resolve()
    evidence[str(norm)] = sha256(norm)
    qwen_path = Path(os.environ.get("QWEN3VL_PATH") or wrapper_config["model"]["tokenizer_path"]).resolve()
    if not (qwen_path / "config.json").is_file():
        raise ValueError("The Qwen model/processor configuration is unavailable")
    for pattern in ("*.json", "*.jinja", "tokenizer.model"):
        for path in sorted(qwen_path.glob(pattern)):
            evidence[str(path)] = sha256(path)
    source = Path(__file__).resolve().parents[1]
    names = set(SOURCE_FILES)
    for package in ("lingbotvla", "deploy", "scripts", "tasks"):
        names.update(str(p.relative_to(source)) for p in (source / package).rglob("*.py"))
    sources = {name: sha256(source / name) for name in sorted(names)}
    output.mkdir(parents=True, exist_ok=False)
    lock = {
        "schema_version": 1,
        "purpose": "Consumed-state feature diagnostic, never final evaluation",
        "wrapper": str(wrapper),
        "norm": str(norm),
        "qwen_path": str(qwen_path),
        "cases": cases,
        "adapters": adapters,
        "expected_base_tensors": 1708,
        "times": [0.05, 0.5, 0.95],
        "noise_seeds": [91001, 91002, 91003, 91004],
        "expected_points": sum(TASK_STATES.values()) * 12,
        "states_by_task": TASK_STATES,
        "diagnostic_action_source": "full generated_action, not a reconstruction of padded training targets",
        "evidence_sha256": evidence,
        "source_sha256": sources,
        "legacy_scope": "Original training branch under common diagnostic TF32 settings, not a replay of old optimizer steps",
    }
    write_json(output / "input_lock.json", lock, immutable=True)
    write_json(output / "empty_registry.json", {"schema_version": 1, "tasks": {}}, immutable=True)
    write_json(
        output / "seal.json",
        {
            "input_lock_sha256": sha256(output / "input_lock.json"),
            "empty_registry_sha256": sha256(output / "empty_registry.json"),
        },
        immutable=True,
    )
    return lock


def require_idle_gpu(gpu, active_study):
    if gpu not in (4, 5, 6, 7) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise ValueError("Use exactly one authorized GPU via CUDA_VISIBLE_DEVICES=4|5|6|7")
    status = json.loads((active_study / "status.json").read_text())
    # Do not inspect outcomes, final result files or task successes. A GPU that
    # is briefly empty between frozen jobs is NOT available for another model.
    if status.get("stage") != "complete" or status.get("state") not in ("significant_uplift", "not_proven"):
        raise RuntimeError("Frozen final study is not complete; refusing to compete for its GPUs")
    line = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    applications = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    )
    if int(line.strip()) > 1024 or applications.strip():
        raise RuntimeError("Authorized GPU is occupied; no waiting/retry/eviction is permitted")
    for key, value in {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NVIDIA_TF32_OVERRIDE": "0",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }.items():
        if os.environ.get(key) != value:
            raise ValueError(f"Set {key}={value} before starting the probe")


def compare_arrays(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Feature shape/dtype mismatch")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Nonfinite feature comparison")
    delta = left.astype(np.float64) - right.astype(np.float64)
    return {
        "byte_identical": left.tobytes() == right.tobytes(),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.abs(delta).max()),
    }


def verify_lock(output):
    import lingbotvla.recap.evaluation as evaluation

    source = Path(__file__).resolve().parents[1]
    if Path(evaluation.__file__).resolve() != source / "lingbotvla/recap/evaluation.py":
        raise ValueError("Set PYTHONPATH to the frozen repair code; never mix source trees")
    seal = json.loads((output / "seal.json").read_text())
    if (
        sha256(output / "input_lock.json") != seal["input_lock_sha256"]
        or sha256(output / "empty_registry.json") != seal["empty_registry_sha256"]
    ):
        raise ValueError("Diagnostic seal changed")
    lock = json.loads((output / "input_lock.json").read_text())
    if (
        lock["states_by_task"] != TASK_STATES
        or lock["times"] != [0.05, 0.5, 0.95]
        or lock["noise_seeds"] != [91001, 91002, 91003, 91004]
        or len(lock["cases"]) * 12 != lock["expected_points"]
        or lock["expected_points"] != sum(TASK_STATES.values()) * 12
    ):
        raise ValueError("Incomplete/changed diagnostic grid")
    from collections import Counter

    identities = {(case["task"], case["seed"]) for case in lock["cases"]}
    if len(identities) != len(lock["cases"]) or dict(Counter(t for t, _ in identities)) != TASK_STATES:
        raise ValueError("Duplicate/missing diagnostic states")
    for path, expected in lock["evidence_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Frozen diagnostic input changed: {path}")
    source = Path(__file__).resolve().parents[1]
    for path, expected in lock["source_sha256"].items():
        if sha256(source / path) != expected:
            raise ValueError(f"Diagnostic source changed: {path}; use the frozen snapshot")
    for case in lock["cases"]:
        ensure_consumed_input(case["manifest"], case["seed"])
    return lock


def base_tensor_hashes(model):
    result = {}
    for name, tensor in model.state_dict().items():
        if name.startswith("recap_"):
            continue
        value = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def run(output, gpu, active_study):
    require_idle_gpu(gpu, active_study)  # BEFORE torch/model imports or CUDA initialization
    if (output / "started.json").exists():
        raise FileExistsError("Probe already started; preserve failures and audit manually")
    lock = verify_lock(output)
    write_json(
        output / "started.json", {"gpu": gpu, "input_lock_sha256": sha256(output / "input_lock.json")}, immutable=True
    )
    import sys

    import torch
    import yaml

    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from lingbotvla.recap.adapter import apply_recap_velocity_lora
    from lingbotvla.recap.training import ADAPTER_PARAMETERS

    source = Path(__file__).resolve().parents[1]
    for name in (
        "deploy.lingbot_vla_v2_policy",
        "lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2",
        "lingbotvla.models.vla.lingbot_vla.qwen2_action_expert",
        "lingbotvla.ops.robby_moe",
    ):
        if Path(sys.modules[name].__file__).resolve() != source / (name.replace(".", "/") + ".py"):
            raise ValueError(f"Imported model code outside the frozen repair snapshot: {name}")
    os.environ["QWEN3VL_PATH"] = lock["qwen_path"]
    policy = LingbotVLAv2Server(
        lock["wrapper"],
        robot_norm_path=lock["norm"],
        use_bf16=False,
        use_fp32=True,
        use_compile=False,
        recap_adapter_registry=str(output / "empty_registry.json"),
        deterministic_algorithms=True,
        recap_condition="null",
    )
    flow = policy.vla.model
    deployed_flags = {
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    flow.config.train_recap_adapter_only = True
    flow.config.recap_prompt_enabled = False
    flow.config.enable_visual_distillation = False
    flow.config.bias_update_speed = flow.config.sequence_wise_loss_coeff = flow.config.router_z_loss_coeff = 0
    flow.config.use_compile = False
    if getattr(policy.data_config, "recap_prompt_enabled", True) or getattr(
        policy.data_config, "image_augment", False
    ):
        raise ValueError("Probe requires literal unconditioned text and disabled image augmentation")
    for name, parameter in flow.named_parameters():
        parameter.requires_grad_(name in ADAPTER_PARAMETERS)
    # No optimizer exists in this script. Hash every frozen base tensor, not
    # just parameters with requires_grad=False or their mutation counters.
    base_before = base_tensor_hashes(flow)
    if len(base_before) != lock["expected_base_tensors"]:
        raise ValueError("Unexpected base tensor inventory")
    moe_blocks = {
        name: module
        for name, module in flow.named_modules()
        if hasattr(module, "num_experts") and hasattr(module, "experts")
    }
    for module in moe_blocks.values():
        module._recap_capture_backend = True
    records, artifacts = [], {}
    for case_index, case in enumerate(lock["cases"]):
        policy.reset("robotwin", task_name=case["task"], environment_seed=case["seed"])
        policy.load_recap_adapter_weights(lock["adapters"][case["task"]]["path"])
        train_config = yaml.safe_load(Path(lock["adapters"][case["task"]]["config"]).read_text())["train"]
        if train_config["action_fp32"] or train_config["loss_type"] != "L1_fm":
            raise ValueError("Probe hook assumes the audited L1_fm/action_fp32=False training recipes")
        with np.load(case["npz"], allow_pickle=False) as arrays:
            raw = {
                k.removeprefix("observation::"): arrays[k].copy()
                for k in arrays.files
                if k.startswith("observation::")
            }
            raw["task"] = case["instruction"]
            generated = arrays["generated_action"].copy()
        deployment = policy._prepare_model_input(copy.deepcopy(raw), recap_condition="null")
        # Use the real feature transform to normalize actions, not a hand-written
        # 14->55 mapping. Verify the training preprocessing has the SAME inputs.
        training_raw = copy.deepcopy(raw)
        transform = policy.vla.feature_transform
        if len(transform.org_features["actions"]) != 1:
            raise ValueError("Expected one RoboTwin action source")
        action_key = transform.org_features["actions"][0]
        training_raw[action_key] = torch.from_numpy(generated)
        training_raw[action_key + "_is_pad"] = torch.zeros(len(generated), dtype=torch.bool)
        training_raw[transform.recap_indicator_key] = "null"
        policy.resize_image(training_raw)
        training_raw = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in training_raw.items()}
        transformed = transform.apply(training_raw, policy_eval=False)
        for key in ("images", "img_masks", "state", "lang_tokens", "lang_masks", "image_grid_thw"):
            compare_arrays(deployment[key].numpy(), transformed[key].numpy())
            if not torch.equal(deployment[key], transformed[key]):
                raise ValueError(f"Training/deployment preprocessing mismatch at {key}")

        def device_tensor(value):
            return value.unsqueeze(0).to(
                device="cuda", dtype=torch.float32 if value.is_floating_point() else value.dtype
            )

        inputs = {
            k: device_tensor(deployment[k])
            for k in ("images", "img_masks", "state", "lang_tokens", "lang_masks", "image_grid_thw")
        }
        actions = device_tensor(transformed["actions"])
        arrays_out = {}
        for t in lock["times"]:
            for noise_seed in lock["noise_seeds"]:
                generator = torch.Generator(device="cpu").manual_seed(noise_seed)
                noise = torch.randn(actions.shape, generator=generator, dtype=torch.float32).cuda()
                time = torch.tensor([t], device="cuda", dtype=torch.float32)
                x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
                captured = {}

                def hook(_module, args, value):
                    captured["hidden"], captured["velocity"] = args[0].detach().clone(), value.detach().clone()

                handle = flow.action_out_proj.register_forward_hook(hook)
                try:
                    flow.config.recap_training_backend = "legacy"
                    flow.config.use_cache = False
                    flow.qwenvl_with_expert.config.attention_implementation = train_config["attention_implementation"]
                    flow.train()
                    for module in moe_blocks.values():
                        module._recap_last_backend = None
                    torch.use_deterministic_algorithms(False)
                    flow(**inputs, actions=actions, noise=noise, time=time, loss_type="L1_fm", recap_condition_id=[-1])
                finally:
                    handle.remove()
                old_h, old_v = captured["hidden"], captured["velocity"]
                legacy_telemetry = {
                    name: copy.deepcopy(module._recap_last_backend) for name, module in moe_blocks.items()
                }
                if not legacy_telemetry or any(
                    row is None or row["backend"] != "fused_bf16" for row in legacy_telemetry.values()
                ):
                    raise ValueError("The original training MoE branch was not actually executed")
                flow.config.recap_training_backend = "deployment"
                flow.config.use_cache = True
                flow.config.attention_implementation = "eager"
                flow.qwenvl_with_expert.config.attention_implementation = "eager"
                flow.config.loss_type = "L1_fm"
                torch.use_deterministic_algorithms(True, warn_only=False)
                h, v = flow.recap_frozen_features(**inputs, x_t=x_t, timestep=time)
                h_repeat, v_repeat = flow.recap_frozen_features(**inputs, x_t=x_t, timestep=time)
                telemetry = copy.deepcopy(flow._recap_backend_telemetry)
                flow.eval()
                # Compare against the actual server flags as loaded, not merely
                # the same functions under the newly tightened training flags.
                torch.set_float32_matmul_precision(deployed_flags["matmul_precision"])
                torch.backends.cuda.matmul.allow_tf32 = deployed_flags["matmul_allow_tf32"]
                torch.backends.cudnn.allow_tf32 = deployed_flags["cudnn_allow_tf32"]
                with torch.no_grad():
                    pad, pos, cache = flow.prepare_velocity_prefix(
                        inputs["images"],
                        inputs["img_masks"],
                        inputs["lang_tokens"],
                        inputs["lang_masks"],
                        inputs["image_grid_thw"],
                    )
                    inference_v = flow.predict_velocity(
                        inputs["state"], pad, cache, x_t, time, prefix_position_ids=pos, recap_condition_id=[-1]
                    )
                    residual = apply_recap_velocity_lora(
                        h,
                        [1],
                        flow.recap_velocity_lora_a,
                        flow.recap_velocity_lora_b,
                        scale=flow.config.recap_adapter_scale,
                        signed_axis=flow.config.recap_signed_velocity_axis,
                    )
                torch.set_float32_matmul_precision("highest")
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False

                def cpu(x):
                    return x.detach().cpu().numpy()

                record = {
                    "case": case_index,
                    "task": case["task"],
                    "seed": case["seed"],
                    "time": t,
                    "noise_seed": noise_seed,
                    "old_vs_deployment_hidden": compare_arrays(cpu(old_h), cpu(h)),
                    "old_vs_deployment_velocity": compare_arrays(cpu(old_v), cpu(v)),
                    "new_vs_inference": compare_arrays(cpu(v), cpu(inference_v)),
                    "repeat_hidden": compare_arrays(cpu(h), cpu(h_repeat)),
                    "repeat_velocity": compare_arrays(cpu(v), cpu(v_repeat)),
                    "positive_residual_rms": float(np.sqrt(np.mean(cpu(residual).astype(float) ** 2))),
                    "backend_telemetry": telemetry,
                    "legacy_backend_telemetry": legacy_telemetry,
                }
                records.append(record)
                key = f"t{t}_n{noise_seed}"
                for name, value in (("old_h", old_h), ("old_v", old_v), ("h", h), ("v", v), ("residual", residual)):
                    arrays_out[key + "_" + name] = cpu(value)
        path = output / f"features_{case_index:03d}.npz"
        from io import BytesIO

        buffer = BytesIO()
        np.savez_compressed(buffer, **arrays_out)
        payload = buffer.getvalue()
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(payload).hexdigest()
        if sha256(path) != digest:
            raise OSError("Feature cache read-back verification failed")
        artifacts[path.name] = digest
        print(json.dumps({"completed_states": case_index + 1, "feature_points": len(records)}), flush=True)
    if base_before != base_tensor_hashes(flow):
        raise ValueError("Frozen base parameter/buffer mutation detected during feature audit")
    passed = len(records) == lock["expected_points"] and all(
        all(row[k]["byte_identical"] for k in ("new_vs_inference", "repeat_hidden", "repeat_velocity"))
        for row in records
    )
    report = {
        "passed": passed,
        "records": records,
        "artifacts_sha256": artifacts,
        "base_tensors_unchanged": True,
        "base_tensor_sha256": base_before,
        "policy_episodes": 0,
        "optimizer_steps": 0,
        "torch": torch.__version__,
        "gpu": gpu,
        "deployed_flags": deployed_flags,
        "input_lock_sha256": sha256(output / "input_lock.json"),
        "scope": lock["purpose"],
        "legacy_scope": lock["legacy_scope"],
    }
    write_json(output / "backend_parity.json", report, immutable=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--wrapper", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(4, 5, 6, 7))
    parser.add_argument("--active-study", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.training_root is None or args.wrapper is None:
            parser.error("--prepare requires --training-root and --wrapper")
        print(
            json.dumps(
                {
                    "prepared_points": prepare(args.training_root.resolve(), args.output, args.wrapper)[
                        "expected_points"
                    ]
                }
            )
        )
    else:
        if args.gpu is None or args.active_study is None:
            parser.error("--run requires --gpu and --active-study")
        try:
            result = run(args.output, args.gpu, args.active_study)
        except Exception as exc:
            # Do not turn an intentional resource refusal into a started probe.
            if (args.output / "started.json").exists() and not (args.output / "failure.json").exists():
                write_json(args.output / "failure.json", {"error": repr(exc)}, immutable=True)
            raise
        if not result["passed"]:
            raise SystemExit("Feature parity gate failed; no training is authorized")


if __name__ == "__main__":
    main()
