#!/usr/bin/env python3
"""TEMP diagnostic: does the trained RECAP adapter fit its own training labels?

Per sampled training decision, computes the masked flow-matching loss of the
executed action chunk under positive/null/negative conditions with paired
noise and a fixed time grid. Fails closed on any per-decision error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.v2 import Resize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lingbotvla.recap.conditioning import recap_condition_id  # noqa: E402
from lingbotvla.recap.decision_preprocess import (  # noqa: E402
    load_decision_observation,
)
from lingbotvla.recap.policy_loader import load_frozen_flow_model  # noqa: E402
from lingbotvla.recap.rollouts import load_rollout_decisions  # noqa: E402
from scripts.recap_open_loop_check import (  # noqa: E402
    load_executed_actions,
    load_label_map,
    select_labeled_decisions,
)


CONDITIONS = ("positive", "null", "negative")
EXCLUDED_TASKS = {"adjust_bottle"}


def build_train_style_inputs(observation, executed, feature_transform, image_size):
    """One apply(policy_eval=False) mirroring the LeRobot training samples."""

    item = dict(observation)
    if getattr(feature_transform, "recap_enabled", False):
        item[feature_transform.recap_indicator_key] = -1
    resize = Resize((int(image_size), int(image_size)))
    for image_key in feature_transform.org_features["images"]:
        array = np.asarray(item[image_key])
        tensor = torch.as_tensor(array).permute(2, 0, 1).contiguous()
        item[image_key] = resize(tensor.to(dtype=torch.float32))

    chunk_size = int(feature_transform.chunk_size)
    if executed.shape[0] > chunk_size:
        executed = executed[:chunk_size]
    chunk = np.empty((chunk_size, executed.shape[1]), dtype=np.float32)
    chunk[: len(executed)] = executed
    chunk[len(executed) :] = executed[-1]
    is_pad = np.zeros(chunk_size, dtype=np.bool_)
    is_pad[len(executed) :] = True
    item["action"] = torch.from_numpy(chunk)
    item["action_is_pad"] = torch.from_numpy(is_pad)
    for key, value in list(item.items()):
        if isinstance(value, np.ndarray):
            item[key] = torch.from_numpy(value)
    transformed = feature_transform.apply(item, policy_eval=False)
    return transformed, torch.from_numpy(is_pad)


def decision_condition_losses(flow_model, transformed, is_pad, *, device, dtype, t_grid, seed):
    images = transformed["images"].unsqueeze(0).to(device=device, dtype=dtype)
    img_masks = transformed["img_masks"].unsqueeze(0).to(device=device, dtype=torch.bool)
    lang_tokens = transformed["lang_tokens"].unsqueeze(0).to(device=device, dtype=torch.long)
    lang_masks = transformed["lang_masks"].unsqueeze(0).to(device=device, dtype=torch.bool)
    state = transformed["state"].unsqueeze(0).to(device=device, dtype=dtype)
    actions = transformed["actions"].unsqueeze(0).to(device=device, dtype=dtype)
    grid_thw = transformed["image_grid_thw"].unsqueeze(0).to(device=device, dtype=torch.long)
    joint_mask = transformed["action_joint_mask"].to(device=device)
    valid_steps = ~is_pad.to(device=device)

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    noise = torch.randn(actions.shape, device=device, dtype=dtype)

    losses_by_condition: dict[str, float] = {}
    with torch.no_grad():
        for condition in CONDITIONS:
            per_t = []
            for t_value in t_grid:
                time = torch.full((1,), float(t_value), device=device, dtype=dtype)
                outputs = flow_model.forward(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                    actions,
                    noise=noise,
                    time=time,
                    loss_type="L1_fm",
                    image_grid_thw=grid_thw,
                    recap_condition_id=torch.tensor(
                        [recap_condition_id(condition)], device=device, dtype=torch.long
                    ),
                )
                elementwise = outputs[0][0]  # [chunk, max_action_dim]
                masked = elementwise[valid_steps][:, joint_mask]
                per_t.append(float(masked.mean().item()))
            losses_by_condition[condition] = float(np.mean(per_t))
    return losses_by_condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--robot-config",
        default=str(PROJECT_ROOT / "configs/robot_configs/robotwin.yaml"),
    )
    parser.add_argument("--samples-per-class", type=int, default=40)
    parser.add_argument("--t-grid", default="0.2,0.5,0.8")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    t_grid = [float(item) for item in args.t_grid.split(",")]
    dtype = torch.float32

    decisions = [
        item
        for item in load_rollout_decisions(args.rollout_dir)
        if item.task_name not in EXCLUDED_TASKS
    ]
    label_map = load_label_map(args.labels)
    selected = select_labeled_decisions(decisions, label_map, args.samples_per_class)

    flow_model, feature_transform, meta = load_frozen_flow_model(
        args.checkpoint, args.robot_config, device=args.device, dtype=dtype
    )
    image_keys = sorted(feature_transform.org_features["images"])
    image_size = int(meta.get("image_size", 256))

    rows = {"positive": [], "negative": []}
    counter = 0
    for class_name in ("positive", "negative"):
        for decision in selected[class_name]:
            context = f"{decision.episode_id}::{decision.decision_index}"
            try:
                observation = load_decision_observation(decision, image_keys)
                executed = load_executed_actions(decision)
                transformed, is_pad = build_train_style_inputs(
                    observation, executed, feature_transform, image_size
                )
                losses = decision_condition_losses(
                    flow_model,
                    transformed,
                    is_pad,
                    device=args.device,
                    dtype=dtype,
                    t_grid=t_grid,
                    seed=args.seed + counter,
                )
            except Exception as exc:
                raise ValueError(f"Failed FM-loss diagnostic on decision {context}: {exc}") from exc
            rows[class_name].append(
                {"task": decision.task_name, "context": context, "losses": losses}
            )
            counter += 1
            if counter % 10 == 0:
                print(f"Scored {counter} decisions ...", flush=True)

    summary = {"samples": {k: len(v) for k, v in rows.items()}, "per_class": {}}
    for class_name, items in rows.items():
        mean_losses = {
            condition: float(np.mean([row["losses"][condition] for row in items]))
            for condition in CONDITIONS
        }
        wins = float(
            np.mean([row["losses"][class_name] < row["losses"]["null"] for row in items])
        )
        opposite = "negative" if class_name == "positive" else "positive"
        beats_opposite = float(
            np.mean([row["losses"][class_name] < row["losses"][opposite] for row in items])
        )
        summary["per_class"][class_name] = {
            "mean_loss": mean_losses,
            "win_rate_matching_vs_null": wins,
            "win_rate_matching_vs_opposite": beats_opposite,
            "mean_gap_null_minus_matching": float(
                np.mean(
                    [row["losses"]["null"] - row["losses"][class_name] for row in items]
                )
            ),
        }
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "t_grid": t_grid,
            "loss_type": "L1_fm",
            "samples_per_class": int(args.samples_per_class),
            "seed": int(args.seed),
        }
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
