#!/usr/bin/env python3
"""Attach Monte-Carlo returns, advantages, and RECAP labels.

``--input`` may be an episode JSONL file or a RECAP rollout directory. A
directory is scored in-process when ``--checkpoint`` is set, so labeling does
not require ``recap_predict_values.py``. Camera observations are loaded from
each decision NPZ when the checkpoint is a visual-task critic.

JSONL input uses one JSON object per episode::

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
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lingbotvla.recap.labels import (  # noqa: E402
    RecapAdvantageLabel,
    compute_advantages,
    label_advantages,
    n_step_advantages,
    positive_quantile_threshold,
    time_to_success_rewards,
)
from lingbotvla.recap.rollouts import (  # noqa: E402
    decisions_to_json_episode,
    group_decisions_by_episode,
    load_rollout_decisions,
)
from lingbotvla.recap.value import monte_carlo_returns  # noqa: E402
from lingbotvla.recap.value_infer import load_value_checkpoint, predict_decision_values  # noqa: E402


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
    n_step: int | None = None,
    n_step_unit: str = "actions",
    force_intervention_positive: bool = True,
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
    durations = None
    if any("executed_action_length" in step for step in steps):
        durations = torch.tensor(
            [float(step.get("executed_action_length") or 1) for step in steps],
            dtype=torch.float32,
        )
    returns = monte_carlo_returns(
        rewards,
        terminated=terminated,
        valid_mask=valid,
        gamma=gamma,
    )
    if n_step is None:
        advantages = compute_advantages(returns, values)
    else:
        advantages = n_step_advantages(
            rewards,
            values,
            n=n_step,
            gamma=gamma,
            terminated=terminated,
            valid_mask=valid,
            durations=durations,
            horizon_unit=n_step_unit,
        )
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

    if force_intervention_positive:
        for index, step in enumerate(steps):
            if valid[index] and bool(step.get("intervention", False)):
                labels[index] = int(RecapAdvantageLabel.POSITIVE)

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


def load_source_episodes(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    """Load episodes from a JSONL file or a recorded rollout directory."""

    source = Path(args.input).expanduser()
    if source.is_dir():
        decisions = load_rollout_decisions(
            source,
            failure_penalty=args.failure_penalty,
            gamma=args.gamma,
        )
        if args.label_mode == "advantage":
            if not args.checkpoint:
                raise ValueError(
                    "Directory input in advantage mode requires --checkpoint; "
                    "use --label-mode outcome to skip the value model"
                )
            values = predict_decision_values(
                decisions,
                args.checkpoint,
                device=args.device,
                batch_size=args.batch_size,
                embedding_cache=args.embedding_cache,
            )
        else:
            values = [0.0] * len(decisions)
        value_by_key = {
            (item.episode_id, item.decision_index): value
            for item, value in zip(decisions, values)
        }
        episodes = []
        for episode_id, items in group_decisions_by_episode(decisions).items():
            episode = decisions_to_json_episode(
                items,
                [value_by_key[(item.episode_id, item.decision_index)] for item in items],
            )
            episodes.append((episode_id, episode))
        return episodes
    if not source.is_file():
        raise FileNotFoundError(f"Input must be a JSONL file or rollout directory, got {source}")
    episodes = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {source}:{line_number}: {exc}") from exc
            identity = str(episode.get("episode_id") or f"{source}:{line_number}")
            episodes.append((identity, episode))
    return episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Episode JSONL, or a RECAP rollout directory containing manifest.json files",
    )
    parser.add_argument("--output", required=True, help="Output labeled episode JSONL")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Value checkpoint used when --input is a rollout directory in advantage mode",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--embedding-cache",
        default=None,
        help="Optional VLM embedding cache for vlm_pooled checkpoints",
    )
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
    parser.add_argument(
        "--n-step",
        type=int,
        default=None,
        help="If set, label with n-step advantages instead of full Monte-Carlo returns minus V.",
    )
    parser.add_argument(
        "--n-step-unit",
        choices=("actions", "decisions"),
        default="actions",
        help=(
            "actions (default) accumulates executed_action_length so n=50 is the "
            "paper's action horizon; decisions counts JSONL/policy chunks."
        ),
    )
    parser.add_argument(
        "--positive-fraction",
        type=float,
        default=None,
        help="If set, per-task advantage quantile so about this fraction of valid steps are positive.",
    )
    parser.add_argument(
        "--force-intervention-positive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force teleoperated intervention steps to the positive RECAP label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint:
        metadata = load_value_checkpoint(args.checkpoint)
        args.failure_penalty = float(metadata["failure_penalty"])
        args.gamma = float(metadata["gamma"])
    thresholds = _load_thresholds(args.threshold_map)
    source_episodes = load_source_episodes(args)
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    episode_count = 0
    step_count = 0
    label_counts = {-1: 0, 0: 0, 1: 0}

    if args.positive_fraction is not None:
        grouped_advantages: dict[str, list[float]] = defaultdict(list)
        for identity, episode in source_episodes:
            try:
                preview = label_episode(
                    episode,
                    value_key=args.value_key,
                    gamma=args.gamma,
                    failure_penalty=args.failure_penalty,
                    threshold=0.0,
                    neutral_margin=0.0,
                    label_mode="advantage",
                    n_step=args.n_step,
                    n_step_unit=args.n_step_unit,
                    force_intervention_positive=False,
                )
            except Exception as exc:
                raise ValueError(f"Failed to label {identity}: {exc}") from exc
            task = str(episode.get("task", ""))
            for step in preview["steps"]:
                if not bool(step.get("valid", True)):
                    continue
                grouped_advantages[task].append(float(step["recap_advantage"]))
        for task, values in grouped_advantages.items():
            thresholds[task] = positive_quantile_threshold(
                torch.tensor(values, dtype=torch.float32),
                positive_fraction=args.positive_fraction,
            )

    with output_path.open("w", encoding="utf-8") as target:
        for identity, episode in source_episodes:
            try:
                task_threshold = thresholds.get(str(episode.get("task", "")), args.threshold)
                labeled = label_episode(
                    episode,
                    value_key=args.value_key,
                    gamma=args.gamma,
                    failure_penalty=args.failure_penalty,
                    threshold=task_threshold,
                    neutral_margin=args.neutral_margin,
                    label_mode=args.label_mode,
                    n_step=args.n_step,
                    n_step_unit=args.n_step_unit,
                    force_intervention_positive=args.force_intervention_positive,
                )
            except Exception as exc:
                raise ValueError(f"Failed to label {identity}: {exc}") from exc
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
