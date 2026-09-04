#!/usr/bin/env python3
"""Predict value baselines for recorded RECAP rollouts and emit episode JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lingbotvla.recap.rollouts import (  # noqa: E402
    decisions_to_json_episode,
    load_decision_state,
    load_rollout_decisions,
)
from lingbotvla.recap.value import StateTaskValueModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _load_checkpoint(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    args = parse_args()
    checkpoint = _load_checkpoint(args.checkpoint)
    if checkpoint.get("model_type") != "state_task_categorical_value":
        raise ValueError(f"Unsupported value checkpoint type: {checkpoint.get('model_type')!r}")

    model = StateTaskValueModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    decisions = load_rollout_decisions(
        args.rollout_dir,
        failure_penalty=float(checkpoint["failure_penalty"]),
        gamma=float(checkpoint["gamma"]),
    )
    state_key = str(checkpoint["state_key"])
    states = np.stack([load_decision_state(item, state_key=state_key) for item in decisions])
    state_tensor = torch.from_numpy(states).float()
    state_tensor = (state_tensor - checkpoint["state_mean"]) / checkpoint["state_std"]
    task_to_id = checkpoint["task_to_id"]
    unknown_tasks = sorted({item.task_name for item in decisions if item.task_name not in task_to_id})
    if unknown_tasks:
        raise ValueError("Value checkpoint does not contain these tasks: " + ", ".join(unknown_tasks))
    task_tensor = torch.tensor([task_to_id[item.task_name] for item in decisions], dtype=torch.long)

    predictions = []
    with torch.no_grad():
        for start in range(0, len(decisions), args.batch_size):
            stop = start + args.batch_size
            prediction = model.expected_value(
                state_tensor[start:stop].to(device),
                task_tensor[start:stop].to(device),
            )
            predictions.extend(prediction.cpu().tolist())

    grouped_decisions = defaultdict(list)
    grouped_predictions = defaultdict(list)
    for decision, prediction in zip(decisions, predictions):
        grouped_decisions[decision.episode_id].append(decision)
        grouped_predictions[decision.episode_id].append(prediction)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        for episode_id in grouped_decisions:
            episode = decisions_to_json_episode(
                grouped_decisions[episode_id],
                grouped_predictions[episode_id],
            )
            output.write(json.dumps(episode, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)
    print(
        json.dumps(
            {
                "episodes": len(grouped_decisions),
                "decisions": len(decisions),
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
