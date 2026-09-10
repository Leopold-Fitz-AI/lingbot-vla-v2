#!/usr/bin/env python3
"""Frozen 3-task x 2-backend x 2-initializer x 3-seed compact training study.

No simulator, final data, candidate selection, automatic retries or promotion.
The sine cell runs the production policy forward/loss/backward. The orthogonal
cell shares ONLY detached post-backbone features; loss/gradient equivalence is
checked against the production cell before accepting this cache optimization.
Both cells perform exactly 50 optimizer updates with four common microbatches.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lingbotvla.recap.ablation import (
    BACKENDS,
    BATCH,
    INITIALIZATION_SEED,
    INITIALIZATIONS,
    SEEDS,
    STEPS,
    TASKS,
    assert_training_sources,
    cached_losses,
    checked_gradients,
    draw_seed,
    sample_schedule,
    summarize_pairs,
    tensor_hash,
)
from lingbotvla.recap.evaluation import sha256, write_json
from scripts.recap_audit_backend_parity import (
    base_tensor_hashes,
    ensure_consumed_input,
    require_idle_gpu,
    validate_raw_chunk,
)


def load(path):
    return json.loads(Path(path).read_text())


def freeze(training_root, parity_root, output):
    import yaml

    if output.exists():
        raise FileExistsError("Use a new study; no overwrite/resume")
    parity = load(parity_root / "backend_parity.json")
    prior = load(parity_root / "input_lock.json")
    if not (
        parity["passed"]
        and parity["base_tensors_unchanged"]
        and len(parity["records"]) == 600
        and len(parity["base_tensor_sha256"]) == 1708
        and parity["input_lock_sha256"] == sha256(parity_root / "input_lock.json")
    ):
        raise ValueError("Full-model numerical prerequisite not established")
    evidence = dict(prior["evidence_sha256"])
    evidence.update(
        {
            str(parity_root / n): sha256(parity_root / n)
            for n in ("backend_parity.json", "input_lock.json", "seal.json")
        }
    )
    for name, h in parity["artifacts_sha256"].items():
        if sha256(parity_root / name) != h:
            raise ValueError("Corrupt parity feature evidence")
    source = Path(__file__).resolve().parents[1]
    for name, h in prior["source_sha256"].items():
        if sha256(source / name) != h:
            raise ValueError(f"Model/preprocessing code differs from passed parity gate: {name}")
    tasks = {}
    for task, (objective, count, gpu) in TASKS.items():
        folder = training_root / "training_v2" / task / objective
        cfg_path, txt = folder / "train_config.yaml", folder / "train.txt"
        cfg = yaml.safe_load(cfg_path.read_text())
        lines = [line.split() for line in txt.read_text().splitlines() if line.strip()]
        if len(lines) != 1 or len(lines[0]) != 2 or lines[0][0] != "robotwin":
            raise ValueError("Expected exactly the original RoboTwin dataset")
        dataset = Path(lines[0][1])
        assert_training_sources([dataset, txt, cfg_path])
        info = load(dataset / "meta/info.json")
        if info["total_frames"] != count or info["total_episodes"] != count:
            raise ValueError("Training dataset row count changed")
        for p in [cfg_path, txt, *sorted(dataset.rglob("*"))]:
            if p.is_file():
                evidence[str(p)] = sha256(p)
        split_root = training_root / "state_splits" / task
        summary_path = split_root / "split_summary.json"
        summary = load(summary_path)
        evidence[str(summary_path)] = sha256(summary_path)
        diagnostic = []
        for split in ("train", "holdout"):
            seen = set()
            for path in sorted((split_root / split).rglob("manifest.json")):
                row = load(path)
                meta = row["metadata"]
                seed = int(meta["paired_environment_seed"])
                ensure_consumed_input(path, seed)
                if seed not in summary[split + "_states"] or meta["task_name"] != task or len(row["steps"]) != 1:
                    raise ValueError("Unexpected diagnostic state/task/chunk")
                npz = path.parent / row["steps"][0]["file"]
                validate_raw_chunk(npz)
                with np.load(npz, allow_pickle=False) as arrays:
                    if arrays["executed_action"].shape != (50, 14):
                        raise ValueError("P0 three-task study requires full executed chunks, not padded targets")
                evidence[str(path)], evidence[str(npz)] = sha256(path), sha256(npz)
                diagnostic.append(
                    {
                        "manifest": str(path),
                        "npz": str(npz),
                        "split": split,
                        "seed": seed,
                        "label": int(row["success"]),
                        "pair_id": meta["paired_match_id"],
                        "instruction": row["task"],
                    }
                )
                seen.add(seed)
            if seen != set(summary[split + "_states"]):
                raise ValueError("Incomplete diagnostic cohort")
        # Check pair membership BEFORE any model is trained.
        summarize_pairs([{**d, "time": 0.5, "noise_seed": 0, "positive_fm": 0.0, "base_fm": 0.0} for d in diagnostic])
        expected = {
            "max_steps": 50,
            "global_batch_size": 4,
            "micro_batch_size": 1,
            "lr": 0.001,
            "loss_type": "L1_fm",
            "weight_decay": 0.0,
            "lr_decay_style": "constant",
            "recap_adapter_rank": 8,
            "recap_adapter_scale": 8.0,
            "recap_signed_velocity_axis": True,
        }
        if any(cfg["train"][k] != v for k, v in expected.items()):
            raise ValueError("Original objective/update recipe changed")
        tasks[task] = {
            "gpu": gpu,
            "objective": objective,
            "config": cfg,
            "dataset": str(dataset),
            "rows": count,
            "diagnostic": diagnostic,
            "schedules": {str(seed): sample_schedule(count, seed) for seed in SEEDS},
        }
    for p, expected in evidence.items():
        if sha256(Path(p)) != expected:
            raise ValueError(f"Source data changed since prerequisite: {p}")
    sources = {
        str(p.relative_to(source)): sha256(p)
        for package in ("lingbotvla", "deploy", "scripts", "tasks")
        for p in (source / package).rglob("*.py")
    }
    sources["configs/robot_configs/robotwin.yaml"] = sha256(source / "configs/robot_configs/robotwin.yaml")
    plan = {
        "schema_version": 1,
        "scope": "Consumed-data P0 attribution; not independent policy validation",
        "tasks": tasks,
        "optimizer_seeds": list(SEEDS),
        "formal_optimizer_seed": SEEDS[0],
        "initialization_seed": INITIALIZATION_SEED,
        "initializations": list(INITIALIZATIONS),
        "backends": list(BACKENDS),
        "optimizer_steps_per_cell": STEPS,
        "batch": BATCH,
        "training_runs": 36,
        "total_optimizer_updates": 1800,
        "policy_episodes": 0,
        "times": prior["times"],
        "diagnostic_noise_seeds": prior["noise_seeds"],
        "parity_report": str(parity_root / "backend_parity.json"),
        "wrapper": prior["wrapper"],
        "norm": prior["norm"],
        "qwen_path": prior["qwen_path"],
        "optimizer": {
            "lr": 0.001,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "foreach": True,
            "fused": False,
        },
        "common_frame": "One device, no auxiliary losses/checkpointing/offload/router updates; explicit shared draws; not original FSDP replay",
        "automatic_retries": 0,
        "automatic_promotion": False,
        "evidence_sha256": evidence,
        "source_sha256": sources,
    }
    plan["cells"] = [
        dict(
            task=task,
            gpu=tasks[task]["gpu"],
            backend=backend,
            initialization=init,
            optimizer_seed=seed,
            initialization_seed=INITIALIZATION_SEED,
            steps=STEPS,
            batch=BATCH,
            micro_batch=1,
            scale=8.0,
            rank=8,
            objective=tasks[task]["objective"],
            residual_weight=tasks[task]["config"]["train"].get("recap_residual_loss_weight", 0.0),
            parameter_dtype="float32",
            matmul_precision="highest",
            tf32=False,
            deterministic=backend == "deployment",
            auxiliary_targets=False,
            checkpointing=False,
            sampler="PCG64 epoch permutation drop_last",
            full_production_forward=init == INITIALIZATIONS[0],
        )
        for task in TASKS
        for seed in SEEDS
        for backend in BACKENDS
        for init in INITIALIZATIONS
    ]
    plan["preprocessing_gate"] = preprocessing_gate(tasks, prior["qwen_path"], prior["norm"])
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan, immutable=True)
    write_json(output / "lock.json", {"plan_sha256": sha256(output / "plan.json")}, immutable=True)
    write_json(output / "empty_registry.json", {"schema_version": 1, "tasks": {}}, immutable=True)
    return plan


def verify(root):
    if sha256(root / "plan.json") != load(root / "lock.json")["plan_sha256"]:
        raise ValueError("Study plan lock changed")
    plan = load(root / "plan.json")
    if plan["optimizer_seeds"] != list(SEEDS) or plan["training_runs"] != 36 or plan["automatic_retries"] != 0:
        raise ValueError("Invalid ablation design")
    code = Path(__file__).resolve().parents[1]
    for name, h in plan["source_sha256"].items():
        if sha256(code / name) != h:
            raise ValueError(f"Frozen executable changed: {name}")
    for p, h in plan["evidence_sha256"].items():
        if sha256(Path(p)) != h:
            raise ValueError(f"Frozen training input changed: {p}")
    return plan


def make_dataset(spec, model_config, processor, norm):
    from lingbotvla.data.vla_data.base_dataset import VLADataset
    from lingbotvla.data.vla_data.utils import FeatureTransform

    source = Path(__file__).resolve().parents[1]
    data_cfg = SimpleNamespace(**spec["config"]["data"])
    transform = FeatureTransform(
        str(source / "configs/robot_configs/robotwin.yaml"),
        data_cfg,
        model_config,
        processor,
        norm_stats_path=norm,
        chunk_size=50,
    )
    return VLADataset(
        spec["dataset"],
        "robotwin",
        data_cfg,
        str(source / "configs/robot_configs"),
        config=model_config,
        processor=processor,
        image_size=(256, 256),
        chunk_size=50,
        feature_transform=transform,
    )


def raw_to_row(case, transform):
    import torch
    from torchvision.transforms.v2 import Resize

    with np.load(case["npz"], allow_pickle=False) as arrays:
        raw = {
            k.removeprefix("observation::"): torch.from_numpy(arrays[k].copy())
            for k in arrays.files
            if k.startswith("observation::")
        }
        raw["action"] = torch.from_numpy(arrays["executed_action"].copy())
    raw.update(
        task=case["instruction"],
        recap_label=torch.tensor([case["label"]]),
        action_is_pad=torch.zeros(50, dtype=torch.bool),
    )
    for key in (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ):
        value = raw[key]
        if value.dtype != torch.uint8 or value.shape[-1] != 3:
            raise ValueError("Unexpected raw image format")
        raw[key] = Resize((256, 256))(value.permute(2, 0, 1).float() / 255.0)
    return transform.apply(raw)


def preprocessing_gate(tasks, qwen_path, norm):
    import importlib.util
    from collections import Counter

    from transformers import AutoProcessor

    path = Path(__file__).resolve().parents[1] / "lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py"
    spec = importlib.util.spec_from_file_location("ablation_configuration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processor = AutoProcessor.from_pretrained(qwen_path)
    result = {}

    def identity(row):
        keys = (
            "images",
            "img_masks",
            "state",
            "lang_tokens",
            "lang_masks",
            "image_grid_thw",
            "actions",
            "joint_mask",
            "action_is_pad",
            "recap_condition_id",
        )
        return tuple((key, tensor_hash(row[key])) for key in keys)

    for task, settings in tasks.items():
        cfg = settings["config"]
        config = module.LingbotVLAV2Config(**{**cfg["model"], **cfg["train"]})
        dataset = make_dataset(settings, config, processor, norm)
        originals = Counter(identity(dataset.getitem(i)) for i in range(len(dataset)))
        diagnostics = Counter(
            identity(raw_to_row(case, dataset.feature_transform))
            for case in settings["diagnostic"]
            if case["split"] == "train" and (settings["objective"] != "positive" or case["label"] == 1)
        )
        if originals != diagnostics or sum(originals.values()) != settings["rows"]:
            raise ValueError(f"Original LeRobot and raw diagnostic preprocessing differ for {task}")
        result[task] = {"rows": settings["rows"], "byte_identical_preprocessing": True}
    return result


def device_batch(row):
    import torch

    return {
        k: v.unsqueeze(0).to("cuda", dtype=torch.float32 if v.is_floating_point() else v.dtype)
        for k, v in row.items()
        if isinstance(v, torch.Tensor)
    }


def fixed_draw(flow, actions, seed):
    import torch

    # Fork BOTH CPU and CUDA RNG. No backend can shift the next data/noise draw.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        noise = torch.randn(actions.shape, dtype=torch.float32, device=actions.device)
        times = flow.sample_time(len(actions), actions.device)
    return noise, times


def set_backend(flow, backend, original):
    import torch

    flow.config.recap_training_backend = backend
    flow.config.use_cache = backend == "deployment"
    flow.config.attention_implementation = "eager" if backend == "deployment" else original
    flow.qwenvl_with_expert.config.attention_implementation = flow.config.attention_implementation
    flow.qwenvl_with_expert.attention_interface = flow.qwenvl_with_expert.get_attention_interface()
    torch.use_deterministic_algorithms(backend == "deployment", warn_only=False)
    flow.train()


def null_gate(a, b, hidden, base, scale, *, initial=False):
    import torch

    from lingbotvla.recap.adapter import apply_recap_velocity_lora

    for condition in (-1, 0, 1) if initial else (-1,):
        residual = apply_recap_velocity_lora(hidden, [condition], a, b, scale=scale, signed_axis=True)
        if torch.count_nonzero(residual) or not torch.equal(base + residual, base):
            raise ValueError("B=0/base or trained Null identity failed")


def save_bytes(path, payload):
    import hashlib

    with path.open("xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    if sha256(path) != hashlib.sha256(payload).hexdigest():
        raise OSError("Read-back mismatch")


def save_adapter(folder, pair, metadata):
    from safetensors.torch import load as safe_load
    from safetensors.torch import save

    tensors = {"model.recap_velocity_lora_" + name: p.detach().cpu().contiguous() for name, p in zip(("a", "b"), pair)}
    path = folder / "recap_adapter.safetensors"
    save_bytes(
        path,
        save(
            tensors,
            metadata={
                "format": "pt",
                "schema": "lingbotvla-recap-adapter-v1",
                "signed_velocity_axis": "true",
                "training_provenance": json.dumps(metadata, sort_keys=True),
            },
        ),
    )
    restored = safe_load(path.read_bytes())
    if any(tensor_hash(restored[k]) != tensor_hash(v) for k, v in tensors.items()):
        raise ValueError("Compact adapter reload differs")
    return {"path": str(path), "sha256": sha256(path)}


def run_task(root, task, active_study):
    if (root / task).exists():
        raise FileExistsError("Task already exists; never duplicate/restart a training worker")
    if (root / "stop.json").exists():
        raise RuntimeError("Study stopped after a worker failure; no new launch")
    gpu = TASKS[task][2]
    require_idle_gpu(gpu, active_study)  # No CUDA initialization before resource check.
    plan = verify(root)
    out = root / task
    out.mkdir(exist_ok=False)
    write_json(
        out / "started.json",
        {"gpu": gpu, "time_unix": time.time(), "plan_sha256": sha256(root / "plan.json")},
        immutable=True,
    )
    import torch

    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from lingbotvla.recap.initialization import initialize_velocity_lora_
    from lingbotvla.recap.loss import masked_action_loss
    from lingbotvla.recap.training import ADAPTER_PARAMETERS, install_adapter_optimizer_guard

    spec, cfg = plan["tasks"][task], plan["tasks"][task]["config"]
    os.environ["QWEN3VL_PATH"] = plan["qwen_path"]
    policy = LingbotVLAv2Server(
        plan["wrapper"],
        robot_norm_path=plan["norm"],
        use_bf16=False,
        use_fp32=True,
        use_compile=False,
        deterministic_algorithms=True,
        recap_adapter_registry=str(root / "empty_registry.json"),
        recap_condition="null",
    )
    policy.reset("robotwin", task_name=task, environment_seed=0)
    model, flow = policy.vla, policy.vla.model
    for key, value in dict(
        train_recap_adapter_only=True,
        recap_prompt_enabled=False,
        enable_visual_distillation=False,
        bias_update_speed=0.0,
        sequence_wise_loss_coeff=0.0,
        router_z_loss_coeff=0.0,
        use_compile=False,
        loss_type="L1_fm",
        recap_adapter_scale=8.0,
        recap_signed_velocity_axis=True,
        recap_residual_loss_weight=cfg["train"].get("recap_residual_loss_weight", 0.0),
    ).items():
        setattr(flow.config, key, value)
    for name, p in flow.named_parameters():
        p.requires_grad_(name in ADAPTER_PARAMETERS)
    for module in flow.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    base = base_tensor_hashes(flow)
    if base != load(plan["parity_report"])["base_tensor_sha256"]:
        raise ValueError("Training base differs from the passed 1708-tensor parity gate")
    dataset = make_dataset(spec, model.config, policy.processor, plan["norm"])
    if len(dataset) != spec["rows"]:
        raise ValueError("Actual training loader changed the row count")
    # Tiny datasets fit in CPU RAM. This is the REAL original LeRobot/FeatureTransform path.
    rows = [dataset.getitem(i) for i in range(len(dataset))]
    batch_hashes = [{k: tensor_hash(v) for k, v in row.items() if isinstance(v, torch.Tensor)} for row in rows]
    for row in rows:
        if row["actions"].shape != (50, 55) or row["action_is_pad"].any() or not torch.isfinite(row["actions"]).all():
            raise ValueError("Changed/nonfinite/padded training chunks")
    write_json(out / "preprocessed_rows.json", batch_hashes, immutable=True)
    blocks = {n: m for n, m in flow.named_modules() if hasattr(m, "num_experts") and hasattr(m, "experts")}
    for block in blocks.values():
        block._recap_capture_backend = True
    weight, scale = flow.config.recap_residual_loss_weight, flow.config.recap_adapter_scale
    trained = {}
    for seed in SEEDS:
        for backend in BACKENDS:
            if (root / "stop.json").exists():
                raise RuntimeError("Another worker failed; no further cells launched")
            group = out / f"{backend}_s{seed}"
            group.mkdir(exist_ok=False)
            set_backend(flow, backend, cfg["train"]["attention_implementation"])
            primary = [flow.recap_velocity_lora_a, flow.recap_velocity_lora_b]
            secondary = [torch.nn.Parameter(torch.empty_like(p)) for p in primary]
            pairs = [primary, secondary]
            inits = [
                initialize_velocity_lora_(*pair, scheme=scheme, seed=INITIALIZATION_SEED, init_std=0.02)
                for pair, scheme in zip(pairs, INITIALIZATIONS)
            ]
            optimizers = [
                torch.optim.AdamW(
                    pair, lr=0.001, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, foreach=True, fused=False
                )
                for pair in pairs
            ]
            for optimizer, pair in zip(optimizers, pairs):
                install_adapter_optimizer_guard(optimizer, pair)
            logs, draws = [], []
            for step, indices in enumerate(spec["schedules"][str(seed)]):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                losses = [[], []]
                for micro, idx in enumerate(indices):
                    batch = device_batch(rows[idx])
                    noise, times = fixed_draw(flow, batch["actions"], draw_seed(task, seed, step, micro))
                    captured = {}

                    def hook(_m, args, result):
                        captured["hidden"], captured["base"] = args[0].detach().clone(), result.detach().clone()

                    handle = flow.action_out_proj.register_forward_hook(hook)
                    try:
                        actual = model(**batch, noise=noise, time=times)[0]
                    finally:
                        handle.remove()
                    hidden, v = captured["hidden"], captured["base"]
                    cached = [
                        cached_losses(
                            hidden,
                            v,
                            batch["actions"],
                            noise,
                            batch["recap_condition_id"],
                            *pair,
                            batch["joint_mask"],
                            batch["action_is_pad"],
                            scale=scale,
                            weight=weight,
                        )[0]
                        for pair in pairs
                    ]
                    if not torch.equal(actual, cached[0]):
                        raise ValueError("Cached loss does not exactly reproduce production forward")
                    if step < 2:
                        ga = torch.autograd.grad(actual, primary, retain_graph=True)
                        gc = torch.autograd.grad(cached[0], primary, retain_graph=True)
                        if any(not torch.equal(a, c) for a, c in zip(ga, gc)):
                            raise ValueError("Cached gradients do not exactly reproduce production backward")
                    if step == 0 and micro == 0:
                        for pair in pairs:
                            null_gate(*pair, hidden, v, scale, initial=True)
                    expected_backend = "fused_bf16" if backend == "legacy" else "robby_deterministic"
                    if not blocks or any(
                        m._recap_last_backend is None or m._recap_last_backend["backend"] != expected_backend
                        for m in blocks.values()
                    ):
                        raise ValueError("Actual training backend differs from frozen cell")
                    (actual / BATCH).backward()
                    (cached[1] / BATCH).backward()
                    losses[0].append(float(actual.detach()))
                    losses[1].append(float(cached[1].detach()))
                    draws.append(
                        {
                            "step": step,
                            "micro": micro,
                            "row": idx,
                            "noise_sha256": tensor_hash(noise),
                            "time_sha256": tensor_hash(times),
                        }
                    )
                if any(p.grad is not None for n, p in flow.named_parameters() if n not in ADAPTER_PARAMETERS):
                    raise ValueError("Frozen base acquired gradients")
                norms = [checked_gradients(pair, step=step) for pair in pairs]
                for optimizer, pair in zip(optimizers, pairs):
                    torch.nn.utils.clip_grad_norm_(pair, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    null_gate(*pair, hidden, v, scale)
                logs.append(
                    {
                        "step": step + 1,
                        "mean_loss": [float(np.mean(x)) for x in losses],
                        "gradient_norms_before_clip": norms,
                    }
                )
                write_json(
                    out / "progress.json",
                    {
                        "task": task,
                        "seed": seed,
                        "backend": backend,
                        "optimizer_steps_per_cell": step + 1,
                        "cells_completed": len(trained),
                        "updated_unix": time.time(),
                    },
                )
            if base_tensor_hashes(flow) != base:
                raise ValueError("Base changed during adapter optimization")
            # Expose EVERY cell/seed. No best-seed choice or discarded model.
            for cell, pair, init in zip(INITIALIZATIONS, pairs, inits):
                folder = group / cell
                folder.mkdir()
                metadata = {
                    "task": task,
                    "backend": backend,
                    "initialization": init,
                    "optimizer_seed": seed,
                    "steps": STEPS,
                    "sample_presentations": STEPS * BATCH,
                    "source_configuration": cfg,
                    "effective_cell": next(
                        c
                        for c in plan["cells"]
                        if c["task"] == task
                        and c["optimizer_seed"] == seed
                        and c["backend"] == backend
                        and c["initialization"] == cell
                    ),
                    "base_tensors_unchanged": True,
                    "verified_base_tensors": 1708,
                    "plan_sha256": sha256(root / "plan.json"),
                    "cache_loss_exact": True,
                    "cache_gradient_gate_exact": True,
                }
                artifact = save_adapter(folder, pair, metadata)
                write_json(
                    folder / "training.json",
                    {**metadata, "artifact": artifact, "updates": logs, "draws": draws},
                    immutable=True,
                )
                trained[f"{backend}_s{seed}/{cell}"] = [p.detach().clone() for p in pair]
            write_json(
                group / "backend_telemetry.json", {n: m._recap_last_backend for n, m in blocks.items()}, immutable=True
            )
            if backend == "deployment":
                old = load(out / f"legacy_s{seed}" / INITIALIZATIONS[0] / "training.json")["draws"]
                if old != draws:
                    raise ValueError("Paired backend cells consumed different data/noise/t schedules")
    # Diagnostic evaluation uses the SAME strict deployment function for all 12 cells.
    set_backend(flow, "deployment", cfg["train"]["attention_implementation"])
    evaluation = {name: [] for name in trained}
    for di, case in enumerate(spec["diagnostic"]):
        batch = device_batch(raw_to_row(case, dataset.feature_transform))
        for t in plan["times"]:
            for ns in plan["diagnostic_noise_seeds"]:
                noise, _ = fixed_draw(flow, batch["actions"], ns)
                times = torch.tensor([t], device="cuda", dtype=torch.float32)
                x = times[:, None, None] * noise + (1 - times[:, None, None]) * batch["actions"]
                args = {
                    k: batch[k]
                    for k in ("images", "img_masks", "state", "lang_tokens", "lang_masks", "image_grid_thw")
                }
                h, v = flow.recap_frozen_features(**args, x_t=x, timestep=times)
                base_error, _ = masked_action_loss(
                    (noise - batch["actions"] - v).abs(),
                    action_dim=55,
                    joint_mask=batch["joint_mask"],
                    action_is_pad=batch["action_is_pad"],
                )
                with torch.no_grad():
                    for name, pair in trained.items():
                        _, fm, _ = cached_losses(
                            h,
                            v,
                            batch["actions"],
                            noise,
                            torch.tensor([1], device="cuda"),
                            *pair,
                            batch["joint_mask"],
                            batch["action_is_pad"],
                            scale=scale,
                            weight=0.0,
                        )
                        evaluation[name].append(
                            {k: case[k] for k in ("split", "seed", "label", "pair_id")}
                            | {"time": t, "noise_seed": ns, "base_fm": float(base_error), "positive_fm": float(fm)}
                        )
        write_json(
            out / "progress.json",
            {
                "task": task,
                "stage": "consumed_diagnostics",
                "manifests_complete": di + 1,
                "manifests_expected": len(spec["diagnostic"]),
                "cells_completed": 12,
                "updated_unix": time.time(),
            },
        )
    summaries = {name: summarize_pairs(rows) for name, rows in evaluation.items()}
    if base_tensor_hashes(flow) != base:
        raise ValueError("Base changed in diagnostics")
    write_json(out / "diagnostic_rows.json", evaluation, immutable=True)
    write_json(
        out / "done.json",
        {
            "complete": True,
            "cells": 12,
            "optimizer_steps": 600,
            "base_tensors_unchanged": True,
            "base_tensor_sha256": base,
            "diagnostics": summaries,
            "scope": plan["scope"],
        },
        immutable=True,
    )
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run-task", choices=TASKS)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--training-root", type=Path)
    p.add_argument("--parity-root", type=Path)
    p.add_argument("--active-study", type=Path)
    args = p.parse_args()
    if args.freeze:
        if not args.training_root or not args.parity_root:
            p.error("--freeze requires training/parity roots")
        plan = freeze(args.training_root, args.parity_root, args.output)
        print(json.dumps({"frozen_runs": plan["training_runs"]}))
    else:
        if not args.active_study:
            p.error("--run-task requires --active-study")
        try:
            run_task(args.output, args.run_task, args.active_study)
        except Exception as exc:
            out = args.output / args.run_task
            if (
                not isinstance(exc, FileExistsError)
                and (out / "started.json").exists()
                and not (out / "failure.json").exists()
            ):
                failure = {"error": repr(exc), "task": args.run_task, "retries": 0}
                write_json(out / "failure.json", failure, immutable=True)
                try:
                    save_bytes(args.output / "stop.json", json.dumps(failure, sort_keys=True).encode())
                except FileExistsError:
                    pass
            raise


if __name__ == "__main__":
    main()
