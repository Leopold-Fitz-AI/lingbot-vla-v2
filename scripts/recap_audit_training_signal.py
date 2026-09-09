#!/usr/bin/env python3
"""CPU-only inventory of V2 training signals; does not access final evaluations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from lingbotvla.recap.evaluation import sha256, write_json


TASKS = ("click_bell", "hanging_mug", "place_can_basket", "stack_bowls_three")


def describe(values):
    x = np.asarray(values, dtype=np.float64)
    return {"count": len(x), "min": float(x.min()), "median": float(np.median(x)),
            "max": float(x.max()), "mean": float(x.mean())} if len(x) else {"count": 0}


def rms(x, y):
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Action comparison requires compatible finite arrays")
    return float(np.sqrt(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)))


def observation_delta(x, y):
    keys = [k for k in x.files if k.startswith("observation::")]
    if not keys or set(keys) != {k for k in y.files if k.startswith("observation::")}:
        raise ValueError("Missing/different observation keys")
    numeric, image, identical = 0., 0., True
    for key in keys:
        a, b = x[key], y[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("Paired observations have incompatible arrays")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite paired observations")
        identical &= a.tobytes() == b.tobytes()
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        if "images" in key:
            image = max(image, float(diff.mean()))
        else:
            numeric = max(numeric, float(diff.max()))
    return {"byte_identical": bool(identical), "numeric_max_abs": numeric, "image_mae_max": image}


def adapter_spectrum(path):
    import torch
    from safetensors.torch import load_file
    tensors = load_file(str(path), device="cpu")
    a = next(v for k, v in tensors.items() if k.endswith("recap_velocity_lora_a")).float()
    b = next(v for k, v in tensors.items() if k.endswith("recap_velocity_lora_b")).float()
    initial = torch.sin(torch.arange(a.numel(), dtype=torch.float32).reshape(a.shape) * .017) * .02
    a, b, initial = a.numpy().astype(float), b.numpy().astype(float), initial.numpy().astype(float)
    init_sv = np.linalg.svd(initial[1], compute_uv=False)
    a_sv = np.linalg.svd(a[1], compute_uv=False)
    product = b[1] @ a[1]
    ba_sv = np.linalg.svd(product, compute_uv=False)
    return {"path": str(path), "sha256": sha256(path), "a_shape": list(a.shape), "b_shape": list(b.shape),
            "initial_a_singular_values": init_sv.tolist(), "trained_a_singular_values": a_sv.tolist(),
            "effective_ba_singular_values": ba_sv[:a.shape[1]].tolist(),
            "initial_a_top2_energy": float(np.sum(init_sv[:2] ** 2) / np.sum(init_sv ** 2)),
            "trained_a_top2_energy": float(np.sum(a_sv[:2] ** 2) / np.sum(a_sv ** 2)),
            "a_relative_change_from_initial": float(np.linalg.norm(a[1] - initial[1]) / np.linalg.norm(initial[1])),
            "effective_ba_frobenius": float(np.linalg.norm(product)),
            "unused_negative_b_frobenius": float(np.linalg.norm(b[0]))}


def audit_training(root):
    import yaml
    evidence, tasks = {}, {}

    def load(path):
        evidence[str(path)] = sha256(path)
        return json.loads(path.read_text())

    for task in TASKS:
        split_root = root / "state_splits" / task
        split = load(split_root / "split_summary.json")
        groups, rows = defaultdict(list), []
        for path in sorted((split_root / "train").rglob("manifest.json")):
            m = load(path)
            md = m["metadata"]
            idx = md["paired_original_decision_index"]
            raw_path = Path(md["paired_original_manifest"])
            raw = load(raw_path)
            npz = path.parent / m["steps"][0]["file"]
            evidence[str(npz)] = sha256(npz)
            with np.load(npz, allow_pickle=False) as arrays:
                generated, executed = arrays["generated_action"], arrays["executed_action"]
                if not np.isfinite(generated).all() or not np.isfinite(executed).all():
                    raise ValueError("Nonfinite selected training actions")
                if not np.array_equal(generated[:len(executed)], executed):
                    raise ValueError("Training action differs from the generated executed prefix")
                n = len(executed)
            remaining = len(raw["steps"]) - idx - 1
            row = {"seed": md["paired_environment_seed"], "pair_id": md["paired_match_id"],
                   "positive": m["success"], "original_decision": idx, "executed_steps": n,
                   "following_policy_decisions": remaining,
                   "continuation_seed": md["continuation_policy_seed"],
                   "source_manifest": str(path), "raw_manifest": str(raw_path)}
            rows.append(row)
            groups[(row["seed"], row["pair_id"])].append((path, m, raw_path, raw))
        pairs = []
        for key, members in sorted(groups.items()):
            if len(members) != 2 or {m[1]["success"] for m in members} != {True, False}:
                raise ValueError("Incomplete causal training pair")
            (p, m, rp, rm), (q, n, rq, rn) = members
            idx = m["metadata"]["paired_original_decision_index"]
            if idx != n["metadata"]["paired_original_decision_index"]:
                raise ValueError("Intervention indices differ")
            same_prefix, prefix_rms = True, []
            for decision in range(idx):
                paths = [a.parent / b["steps"][decision]["file"] for a, b in ((rp, rm), (rq, rn))]
                for file in paths:
                    evidence[str(file)] = sha256(file)
                with np.load(paths[0], allow_pickle=False) as x, np.load(paths[1], allow_pickle=False) as y:
                    prefix_rms.append(rms(x["generated_action"], y["generated_action"]))
                    same_prefix &= x["generated_action"].tobytes() == y["generated_action"].tobytes()
            with np.load(p.parent / m["steps"][0]["file"], allow_pickle=False) as x, np.load(q.parent / n["steps"][0]["file"], allow_pickle=False) as y:
                delta = observation_delta(x, y)
                action_rms = rms(x["generated_action"], y["generated_action"])
            pairs.append({"seed": key[0], "pair_id": key[1], "same_instruction": m["task"] == n["task"],
                          "same_continuation": m["metadata"]["continuation_policy_seed"] == n["metadata"]["continuation_policy_seed"],
                          "prefix_actions_identical": bool(same_prefix), "prefix_action_rms": max(prefix_rms, default=0.),
                          "intervention_generated_action_rms": action_rms, "intervention_observation": delta})
        configs, datasets, spectra = {}, {}, {}
        for objective in ("positive", "signed", "regularized"):
            folder = root / "training_v2" / task / objective
            cfg_path = folder / "train_config.yaml"
            evidence[str(cfg_path)] = sha256(cfg_path)
            cfg = yaml.safe_load(cfg_path.read_text())
            configs[objective] = {k: cfg["train"].get(k) for k in (
                "max_steps", "lr", "global_batch_size", "micro_batch_size", "loss_type",
                "recap_residual_loss_weight", "enable_fp32", "enable_mixed_precision", "enable_full_determinism",
                "recap_signed_velocity_axis", "recap_adapter_rank", "recap_adapter_scale", "recap_adapter_init_std", "train_recap_adapter_only")}
            text = folder / "train.txt"
            evidence[str(text)] = sha256(text)
            dataset = Path(text.read_text().split()[1])
            datasets[objective] = load(dataset / "meta/recap_conversion.json")
            labels_path = Path(datasets[objective]["labels"])
            evidence[str(labels_path)] = sha256(labels_path)
            labels = Counter((step["recap_label"], step["recap_advantage"], step["recap_return"], step["value"])
                             for line in labels_path.read_text().splitlines() if line.strip()
                             for step in json.loads(line)["steps"])
            datasets[objective]["label_value_inventory"] = [
                {"label": key[0], "advantage": key[1], "return": key[2], "value": key[3], "count": count}
                for key, count in sorted(labels.items())]
            info = load(dataset / "meta/info.json")
            datasets[objective]["parquet_feature_names"] = sorted(info["features"])
            spectra[objective] = adapter_spectrum(folder / "adapter/recap_adapter.safetensors")
        tasks[task] = {"split": split, "training_rows": rows, "paired_prefix_audits": pairs,
                       "configs": configs, "datasets": datasets, "spectra": spectra,
                       "original_decisions": dict(Counter(str(r["original_decision"]) for r in rows)),
                       "continuation_seeds": dict(Counter(str(r["continuation_seed"]) for r in rows)),
                       "pair_count": len(pairs), "exact_prefix_pairs": sum(p["prefix_actions_identical"] for p in pairs),
                       "exact_intervention_observation_pairs": sum(p["intervention_observation"]["byte_identical"] for p in pairs),
                       "prefix_rms": describe([p["prefix_action_rms"] for p in pairs]),
                       "intervention_rms": describe([p["intervention_generated_action_rms"] for p in pairs]),
                       "positive_execution_lengths": describe([r["executed_steps"] for r in rows if r["positive"]]),
                       "negative_execution_lengths": describe([r["executed_steps"] for r in rows if not r["positive"]]),
                       "positive_following_decisions": describe([r["following_policy_decisions"] for r in rows if r["positive"]])}
    return {"purpose": "Exploratory V2 training-signal/code audit; no final evaluation access or new rollouts",
            "tasks": tasks, "evidence_sha256": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-adapter", type=Path)
    args = parser.parse_args()
    result = audit_training(args.training_root)
    if args.reference_adapter:
        result["reference_adapter"] = adapter_spectrum(args.reference_adapter)
    result["auditor_source_sha256"] = sha256(Path(__file__))
    write_json(args.output, result, immutable=True)
    print(args.output)


if __name__ == "__main__":
    main()
