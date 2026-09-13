#!/usr/bin/env python3
"""Open-loop sanity check for a RECAP velocity adapter's advantage axis.

For labeled rollout decisions, the adapter-trained policy is rolled out under
the positive, null, and negative conditions with identical flow noise (paired
seeds). Predicted action chunks are unnormalized exactly like deployment and
compared against the executed action prefix. A healthy adapter shows the
matching condition closest to the executed actions: on positive-labeled
decisions ``positive < null < negative`` error; on negative-labeled decisions
``negative < null < positive``.

This script does not train and does not modify any deployment path.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lingbotvla.recap.conditioning import recap_condition_id  # noqa: E402
from lingbotvla.recap.decision_preprocess import (  # noqa: E402
    decision_to_prefix_inputs,
    load_decision_observation,
)
from lingbotvla.recap.policy_loader import load_frozen_flow_model  # noqa: E402
from lingbotvla.recap.rollouts import RolloutDecision, load_rollout_decisions  # noqa: E402


CONDITIONS = ("positive", "null", "negative")
CLASS_BY_LABEL = {1: "positive", 0: "negative"}
EXPECTED_ORDER = {
    "positive": ("positive", "null", "negative"),
    "negative": ("negative", "null", "positive"),
}


def _decision_context(decision: RolloutDecision) -> str:
    return f"{decision.episode_id}::{decision.decision_index}"


def load_label_map(labels_path: str | Path) -> dict[tuple[str, int], int]:
    """Parse labeled-episode JSONL into ``(episode_id, decision_index) -> recap_label``."""

    path = Path(labels_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Labels JSONL does not exist: {path}")
    mapping: dict[tuple[str, int], int] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            episode_id = episode.get("episode_id")
            steps = episode.get("steps")
            if not episode_id or not isinstance(steps, list):
                raise ValueError(
                    f"Line {line_number} of {path} must contain episode_id and a steps list"
                )
            for step in steps:
                if "decision_index" not in step or "recap_label" not in step:
                    raise ValueError(
                        f"Line {line_number} of {path} ({episode_id}): every step needs "
                        "decision_index and recap_label"
                    )
                key = (str(episode_id), int(step["decision_index"]))
                label = int(step["recap_label"])
                if label not in (-1, 0, 1):
                    raise ValueError(
                        f"Invalid recap_label {label} for {episode_id}::{step['decision_index']} "
                        f"on line {line_number} of {path}"
                    )
                if key in mapping and mapping[key] != label:
                    raise ValueError(f"Conflicting labels for {key[0]}::{key[1]} in {path}")
                mapping[key] = label
    return mapping


def select_labeled_decisions(
    decisions: Sequence[RolloutDecision],
    label_map: dict[tuple[str, int], int],
    samples_per_class: int,
) -> dict[str, list[RolloutDecision]]:
    """Pick up to ``samples_per_class`` decisions per class, round-robin over tasks."""

    if samples_per_class <= 0:
        raise ValueError(f"samples_per_class must be positive, got {samples_per_class}")
    by_class_task: dict[str, dict[str, list[RolloutDecision]]] = {
        "positive": defaultdict(list),
        "negative": defaultdict(list),
    }
    for decision in decisions:
        key = (decision.episode_id, int(decision.decision_index))
        if key not in label_map:
            raise KeyError(
                f"Missing label for decision {_decision_context(decision)} in the labels file"
            )
        class_name = CLASS_BY_LABEL.get(label_map[key])
        if class_name is not None:
            by_class_task[class_name][decision.task_name].append(decision)

    selected: dict[str, list[RolloutDecision]] = {}
    for class_name, by_task in by_class_task.items():
        tasks = sorted(by_task)
        picked: list[RolloutDecision] = []
        index = 0
        while len(picked) < samples_per_class and any(by_task[task] for task in tasks):
            task = tasks[index % len(tasks)]
            if by_task[task]:
                picked.append(by_task[task].pop(0))
            index += 1
        if not picked:
            raise ValueError(f"No {class_name}-labeled decisions found in the rollout set")
        selected[class_name] = picked
    return selected


def load_executed_actions(decision: RolloutDecision) -> np.ndarray:
    context = _decision_context(decision)
    with np.load(decision.observation_path, allow_pickle=False) as archive:
        if "executed_action" not in archive.files:
            raise KeyError(
                f"Decision {context}: executed_action is absent from {decision.observation_path}"
            )
        executed = np.asarray(archive["executed_action"], dtype=np.float32)
    if executed.ndim != 2:
        raise ValueError(
            f"Decision {context}: executed_action must be [T, action_dim], got {executed.shape}"
        )
    if not np.isfinite(executed).all():
        raise ValueError(f"Decision {context}: executed_action contains NaN or infinity")
    return executed


def unnormalize_actions(
    feature_transform: Any,
    prefix_inputs: dict[str, torch.Tensor],
    actions: torch.Tensor,
    *,
    context: str,
) -> np.ndarray:
    """Invert deployment normalization; concatenate raw action keys in order."""

    item = dict(prefix_inputs)
    item["actions"] = actions.squeeze(0).to(dtype=torch.float32, device="cpu")
    data = feature_transform.unapply(item)
    parts = []
    for key in feature_transform.org_features["actions"]:
        if key not in data:
            raise KeyError(f"Decision {context}: unapply did not return action key {key!r}")
        value = data[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        parts.append(np.asarray(value, dtype=np.float32))
    raw = np.concatenate(parts, axis=-1)
    if raw.ndim != 2:
        raise ValueError(f"Decision {context}: raw actions must be [chunk, dim], got {raw.shape}")
    if not np.isfinite(raw).all():
        raise FloatingPointError(f"Decision {context}: non-finite policy actions after unapply")
    return raw


def compare_decision_conditions(
    flow_model: Any,
    feature_transform: Any,
    decision: RolloutDecision,
    *,
    image_keys: Sequence[str],
    image_size: int,
    compare_steps: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, Any]:
    """Run paired positive/null/negative rollouts for one decision."""

    context = _decision_context(decision)
    observation = load_decision_observation(decision, image_keys)
    prefix_inputs = decision_to_prefix_inputs(observation, feature_transform, image_size)
    executed = load_executed_actions(decision)
    steps = min(int(compare_steps), executed.shape[0])
    if steps <= 0:
        raise ValueError(f"Decision {context}: no executed actions to compare against")

    batched = {
        "images": prefix_inputs["images"].unsqueeze(0).to(device=device, dtype=dtype),
        "img_masks": prefix_inputs["img_masks"].unsqueeze(0).to(device=device, dtype=torch.bool),
        "lang_tokens": prefix_inputs["lang_tokens"].unsqueeze(0).to(device=device, dtype=torch.long),
        "lang_masks": prefix_inputs["lang_masks"].unsqueeze(0).to(device=device, dtype=torch.bool),
        "state": prefix_inputs["state"].unsqueeze(0).to(device=device, dtype=dtype),
        "image_grid_thw": prefix_inputs["image_grid_thw"]
        .unsqueeze(0)
        .to(device=device, dtype=torch.long),
    }

    raw_actions: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for condition in CONDITIONS:
            # Pair the flow noise so only the condition varies across rollouts.
            torch.manual_seed(seed)
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            actions = flow_model.sample_actions(
                batched["images"],
                batched["img_masks"],
                batched["lang_tokens"],
                batched["lang_masks"],
                batched["state"],
                image_grid_thw=batched["image_grid_thw"],
                recap_condition_id=torch.tensor(
                    [recap_condition_id(condition)], device=device, dtype=torch.long
                ),
            )
            raw_actions[condition] = unnormalize_actions(
                feature_transform, prefix_inputs, actions, context=context
            )

    errors = {
        condition: float(
            np.abs(raw_actions[condition][:steps] - executed[:steps]).mean()
        )
        for condition in CONDITIONS
    }
    residual = float(
        np.linalg.norm(raw_actions["positive"][:steps] - raw_actions["null"][:steps], axis=-1).mean()
    )
    return {"errors": errors, "residual_positive_null": residual, "steps": steps}


def aggregate_results(
    rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Aggregate per-decision comparison rows into the ordering summary."""

    per_class: dict[str, Any] = {}
    for class_name, items in rows.items():
        if not items:
            raise ValueError(f"No comparison rows for class {class_name!r}")
        mean_errors = {
            condition: float(np.mean([row["errors"][condition] for row in items]))
            for condition in CONDITIONS
        }
        win_rate = float(
            np.mean([row["errors"][class_name] < row["errors"]["null"] for row in items])
        )
        expected = EXPECTED_ORDER[class_name]
        ordering_pass = (
            mean_errors[expected[0]] < mean_errors[expected[1]] < mean_errors[expected[2]]
        )
        per_class[class_name] = {
            "samples": len(items),
            "mean_error": mean_errors,
            "matching_condition_win_rate_vs_null": win_rate,
            "expected_error_order": list(expected),
            "ordering_pass": bool(ordering_pass),
        }
    residual = float(
        np.mean(
            [row["residual_positive_null"] for items in rows.values() for row in items]
        )
    )
    return {
        "per_class": per_class,
        "residual_magnitude_positive_minus_null": residual,
        "ordering_pass": bool(all(item["ordering_pass"] for item in per_class.values())),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Adapter-trained hf_ckpt directory")
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--labels", required=True, help="Labeled episode JSONL")
    parser.add_argument(
        "--robot-config",
        default=str(PROJECT_ROOT / "configs/robot_configs/robotwin.yaml"),
    )
    parser.add_argument("--samples-per-class", type=int, default=60)
    parser.add_argument("--compare-steps", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.compare_steps <= 0:
        raise ValueError(f"--compare-steps must be positive, got {args.compare_steps}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is not available")
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32

    decisions = load_rollout_decisions(args.rollout_dir)
    label_map = load_label_map(args.labels)
    selected = select_labeled_decisions(decisions, label_map, args.samples_per_class)

    flow_model, feature_transform, meta = load_frozen_flow_model(
        args.checkpoint,
        args.robot_config,
        device=device,
        dtype=dtype,
    )
    image_keys = sorted(feature_transform.org_features["images"])
    image_size = int(meta.get("image_size", 256))

    rows: dict[str, list[dict[str, Any]]] = {"positive": [], "negative": []}
    decision_counter = 0
    for class_name in ("positive", "negative"):
        for decision in selected[class_name]:
            row = compare_decision_conditions(
                flow_model,
                feature_transform,
                decision,
                image_keys=image_keys,
                image_size=image_size,
                compare_steps=args.compare_steps,
                device=device,
                dtype=dtype,
                seed=args.seed + decision_counter,
            )
            rows[class_name].append(row)
            decision_counter += 1
            if decision_counter % 10 == 0:
                print(f"Compared {decision_counter} decisions ...", flush=True)

    summary = aggregate_results(rows)
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "rollout_dir": str(args.rollout_dir),
            "labels": str(args.labels),
            "samples_per_class": int(args.samples_per_class),
            "compare_steps": int(args.compare_steps),
            "seed": int(args.seed),
            "device": device,
            "dtype": str(dtype),
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
