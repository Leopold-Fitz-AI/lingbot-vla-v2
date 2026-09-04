#!/usr/bin/env python3
"""Attach Monte-Carlo returns, advantages, and RECAP labels to rollout JSONL.

Input uses one JSON object per episode::

    {
      "episode_id": "task-a/0001",
      "task": "pick up the cup",
      "success": true,
      "steps": [
        {"value": -12.4, "reward": -1.0, ...},
        {"value": -11.1, "reward": 0.0, ...}
      ]
    }

``reward`` is optional. If every step omits it, time-to-success rewards are
constructed from the episode success label. Value estimates are required and
should come from the behavior-policy value model used for this RECAP round.
The output preserves every field and adds ``recap_return``,
``recap_advantage``, and ``recap_label`` to each step. Labels are 1 (positive),
0 (negative), and -1 (neutral/null branch).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lingbotvla.recap.labels import compute_advantages, label_advantages, time_to_success_rewards  # noqa: E402
from lingbotvla.recap.value import monte_carlo_returns  # noqa: E402


def _load_thresholds(path: str | None) -> dict[str, float]:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("threshold map must be a JSON object mapping task names to numbers")
    return {str(key): float(value) for key, value in data.items()}


def label_episode(
    episode: dict[str, Any],
    *,
    value_key: str = "value",
    gamma: float = 1.0,
    failure_penalty: float = 1000.0,
    threshold: float = 0.0,
    neutral_margin: float = 0.0,
    label_mode: str = "advantage",
) -> dict[str, Any]:
    """Return a copy of one episode with per-step RECAP labels.

    ``advantage`` is the standard RECAP mode. ``outcome`` is a conservative
    fallback for small pilots whose value model is not sufficiently accurate:
    every valid step from a successful episode is positive and every valid
    step from a failed episode is negative. Invalid steps remain neutral.
    Returns and advantages are retained as diagnostics in both modes.
    """

    if label_mode not in {"advantage", "outcome"}:
        raise ValueError(f"Unsupported label mode: {label_mode!r}")

    steps = episode.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Episode {episode.get('episode_id', '<unknown>')!r} has no steps")
    try:
        values = torch.tensor([float(step[value_key]) for step in steps], dtype=torch.float32)
    except KeyError as exc:
        if label_mode == "outcome":
            # Outcome-grounded labels deliberately do not depend on a value
            # model. Zero is an explicit diagnostic placeholder when no value
            # prediction was produced for this conservative fallback mode.
            values = torch.zeros(len(steps), dtype=torch.float32)
        else:
            raise ValueError(
                f"Episode {episode.get('episode_id', '<unknown>')!r} is missing step value key {value_key!r}"
            ) from exc

    has_reward = ["reward" in step for step in steps]
    if any(has_reward) and not all(has_reward):
        raise ValueError("Either every step or no step must provide reward")
    if all(has_reward):
        rewards = torch.tensor([float(step["reward"]) for step in steps], dtype=torch.float32)
    else:
        if "success" not in episode:
            raise ValueError("An episode without step rewards must provide a success label")
        rewards = time_to_success_rewards(
            len(steps),
            success=bool(episode["success"]),
            failure_penalty=failure_penalty,
        )

    terminated = torch.tensor(
        [bool(step.get("terminated", index == len(steps) - 1)) for index, step in enumerate(steps)],
        dtype=torch.bool,
    )
    valid = torch.tensor([bool(step.get("valid", True)) for step in steps], dtype=torch.bool)
    returns = monte_carlo_returns(
        rewards,
        terminated=terminated,
        valid_mask=valid,
        gamma=gamma,
    )
    advantages = compute_advantages(returns, values)
    if label_mode == "advantage":
        labels = label_advantages(
            advantages,
            threshold=threshold,
            neutral_margin=neutral_margin,
            valid_mask=valid,
        )
    else:
        if "success" not in episode:
            raise ValueError("Outcome label mode requires an episode success label")
        outcome_label = 1 if bool(episode["success"]) else 0
        labels = torch.full((len(steps),), outcome_label, dtype=torch.int8)
        labels[~valid] = -1

    output = dict(episode)
    output_steps = []
    for step, value, empirical_return, advantage, label in zip(
        steps,
        values.tolist(),
        returns.tolist(),
        advantages.tolist(),
        labels.tolist(),
    ):
        output_step = dict(step)
        output_step.setdefault(value_key, float(value))
        output_step["recap_return"] = float(empirical_return)
        output_step["recap_advantage"] = float(advantage)
        output_step["recap_label"] = int(label)
        output_steps.append(output_step)
    output["steps"] = output_steps
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input episode JSONL")
    parser.add_argument("--output", required=True, help="Output labeled episode JSONL")
    parser.add_argument("--value-key", default="value", help="Per-step scalar value field")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--failure-penalty", type=float, default=1000.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--threshold-map",
        help="Optional JSON object mapping exact task strings to task-specific thresholds",
    )
    parser.add_argument("--neutral-margin", type=float, default=0.0)
    parser.add_argument(
        "--label-mode",
        choices=("advantage", "outcome"),
        default="advantage",
        help=(
            "Use value-model advantages (default), or conservatively ground "
            "all valid labels in the final episode outcome"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = _load_thresholds(args.threshold_map)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episode_count = 0
    step_count = 0
    label_counts = {-1: 0, 0: 0, 1: 0}
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
                task_threshold = thresholds.get(str(episode.get("task", "")), args.threshold)
                labeled = label_episode(
                    episode,
                    value_key=args.value_key,
                    gamma=args.gamma,
                    failure_penalty=args.failure_penalty,
                    threshold=task_threshold,
                    neutral_margin=args.neutral_margin,
                    label_mode=args.label_mode,
                )
            except Exception as exc:
                raise ValueError(f"Failed to label {input_path}:{line_number}: {exc}") from exc
            target.write(json.dumps(labeled, ensure_ascii=False) + "\n")
            episode_count += 1
            step_count += len(labeled["steps"])
            for step in labeled["steps"]:
                label_counts[int(step["recap_label"])] += 1

    print(
        json.dumps(
            {
                "episodes": episode_count,
                "steps": step_count,
                "label_mode": args.label_mode,
                "labels": {
                    "neutral": label_counts[-1],
                    "negative": label_counts[0],
                    "positive": label_counts[1],
                },
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
