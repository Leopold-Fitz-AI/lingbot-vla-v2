#!/usr/bin/env python3
"""Predict value baselines for recorded RECAP rollouts and emit episode JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lingbotvla.recap.rollouts import decisions_to_json_episode, load_rollout_decisions  # noqa: E402
from lingbotvla.recap.value_infer import predict_decision_values  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--embedding-cache",
        default=None,
        help="Override the VLM embedding cache stored in a vlm_pooled checkpoint.",
    )
    return parser.parse_args()


def _load_checkpoint_meta(path: str) -> dict:
    from lingbotvla.recap.value_infer import load_value_checkpoint

    return load_value_checkpoint(path)


def main() -> None:
    args = parse_args()
    checkpoint = _load_checkpoint_meta(args.checkpoint)
    decisions = load_rollout_decisions(
        args.rollout_dir,
        failure_penalty=float(checkpoint["failure_penalty"]),
        gamma=float(checkpoint["gamma"]),
    )
    predictions = predict_decision_values(
        decisions,
        args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        embedding_cache=args.embedding_cache,
    )

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
