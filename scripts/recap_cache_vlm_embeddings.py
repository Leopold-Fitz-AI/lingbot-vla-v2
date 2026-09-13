#!/usr/bin/env python3
"""Cache frozen VLM prefix embeddings for RECAP value training.

This script does not train. It runs a prefix encoder once per recorded
decision, masked-mean pools the hidden states, and writes a compact cache
consumed by ``train_recap_value.py --encoder vlm_pooled``.

Production loads a frozen LingBot policy checkpoint and encodes with
:func:`lingbotvla.recap.vlm_pool.encode_policy_prefix_hidden`. Tests and
custom encoders inject an encoder through :func:`cache_rollout_embeddings`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lingbotvla.recap.decision_preprocess import (  # noqa: E402
    decision_to_prefix_inputs,
    load_decision_observation,
)
from lingbotvla.recap.policy_loader import load_frozen_flow_model  # noqa: E402
from lingbotvla.recap.rollouts import RolloutDecision, load_rollout_decisions  # noqa: E402
from lingbotvla.recap.value import masked_mean_pool  # noqa: E402
from lingbotvla.recap.vlm_pool import (  # noqa: E402
    encode_policy_prefix_hidden,
    write_vlm_embedding_cache,
)


EncodeFn = Callable[[Any], tuple[torch.Tensor, torch.Tensor]]


def cache_rollout_embeddings(
    rollout_dir: str | Path,
    output: str | Path,
    encode_fn: EncodeFn,
    *,
    failure_penalty: float = 1000.0,
    gamma: float = 1.0,
) -> dict[str, Any]:
    """Encode every decision with ``encode_fn(decision) -> (hidden, mask)``."""

    decisions = load_rollout_decisions(
        rollout_dir,
        failure_penalty=failure_penalty,
        gamma=gamma,
    )
    pooled = []
    for decision in decisions:
        hidden, mask = encode_fn(decision)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
            mask = None if mask is None else mask.unsqueeze(0)
        vector = masked_mean_pool(hidden, mask)
        if vector.shape[0] != 1:
            raise ValueError("encode_fn must return a single-decision embedding")
        pooled.append(vector.reshape(-1).detach().cpu().float())
    embeddings = torch.stack(pooled, dim=0)
    write_vlm_embedding_cache(
        output,
        embeddings=embeddings,
        episode_ids=[item.episode_id for item in decisions],
        decision_indices=[item.decision_index for item in decisions],
    )
    return {
        "decisions": len(decisions),
        "hidden_size": int(embeddings.shape[1]),
        "output": str(Path(output).expanduser()),
    }


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name not in ("cpu", "cuda"):
        raise ValueError(f"--device must be auto, cpu, or cuda, got {name!r}")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is not available")
    return name


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _stack_prefix_batch(items: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack single-decision prefix inputs, padding variable-length language."""

    images = torch.stack([item["images"] for item in items], dim=0)
    img_masks = torch.stack([item["img_masks"] for item in items], dim=0)
    image_grid_thw = torch.stack([item["image_grid_thw"] for item in items], dim=0)

    token_lengths = [int(item["lang_tokens"].shape[0]) for item in items]
    max_length = max(token_lengths)
    lang_tokens = torch.zeros(
        len(items), max_length, dtype=items[0]["lang_tokens"].dtype
    )
    lang_masks = torch.zeros(len(items), max_length, dtype=torch.bool)
    for row, item in enumerate(items):
        length = token_lengths[row]
        lang_tokens[row, :length] = item["lang_tokens"]
        lang_masks[row, :length] = item["lang_masks"].to(dtype=torch.bool)
    return {
        "images": images,
        "img_masks": img_masks,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "image_grid_thw": image_grid_thw,
    }


def _encode_prefix_batch(
    flow_model: Any,
    batch: dict[str, torch.Tensor],
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    hidden, pad_mask = encode_policy_prefix_hidden(
        flow_model,
        batch["images"].to(device=device, dtype=dtype),
        batch["img_masks"].to(device=device, dtype=torch.bool),
        batch["lang_tokens"].to(device=device, dtype=torch.long),
        batch["lang_masks"].to(device=device),
        batch["image_grid_thw"].to(device=device, dtype=torch.long),
    )
    return masked_mean_pool(hidden, pad_mask).detach().cpu().float()


def _prepare_prefix_inputs(
    decision: RolloutDecision,
    feature_transform: Any,
    image_keys: Sequence[str],
    image_size: int,
) -> dict[str, torch.Tensor]:
    context = f"{decision.episode_id}::{decision.decision_index}"
    try:
        observation = load_decision_observation(decision, image_keys)
        return decision_to_prefix_inputs(observation, feature_transform, image_size)
    except Exception as exc:
        raise ValueError(f"Failed to preprocess decision {context}: {exc}") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy-checkpoint", default=None)
    parser.add_argument("--failure-penalty", type=float, default=1000.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--robot-config",
        default=str(PROJECT_ROOT / "configs/robot_configs/robotwin.yaml"),
    )
    parser.add_argument("--norm-stats-path", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument(
        "--image-keys",
        nargs="+",
        default=None,
        help="Explicit NPZ camera keys; defaults to the robot config origin image keys.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.policy_checkpoint:
        raise ValueError(
            "Caching VLM embeddings from rollouts requires --policy-checkpoint. "
            "Tests and custom encoders should call cache_rollout_embeddings()."
        )
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")

    decisions = load_rollout_decisions(
        args.rollout_dir,
        failure_penalty=args.failure_penalty,
        gamma=args.gamma,
    )
    device = _resolve_device(args.device)
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    flow_model, feature_transform, meta = load_frozen_flow_model(
        args.policy_checkpoint,
        args.robot_config,
        device=device,
        dtype=dtype,
        norm_stats_path=args.norm_stats_path,
    )
    image_keys = args.image_keys or sorted(feature_transform.org_features["images"])
    image_size = int(meta.get("image_size", 256))

    pooled: list[torch.Tensor] = []
    pending: list[dict[str, torch.Tensor]] = []
    for decision in decisions:
        pending.append(
            _prepare_prefix_inputs(decision, feature_transform, image_keys, image_size)
        )
        if len(pending) >= args.batch_size:
            pooled.append(
                _encode_prefix_batch(flow_model, _stack_prefix_batch(pending), device=device, dtype=dtype)
            )
            pending = []
    if pending:
        pooled.append(
            _encode_prefix_batch(flow_model, _stack_prefix_batch(pending), device=device, dtype=dtype)
        )
    embeddings = torch.cat(pooled, dim=0)

    extra: dict[str, Any] = {
        "policy_checkpoint": str(meta["policy_checkpoint"]),
        "robot_config": str(meta["robot_config"]),
        "dtype": str(dtype),
        "num_decisions": len(decisions),
    }
    commit = _git_commit()
    if commit:
        extra["git_commit"] = commit
    write_vlm_embedding_cache(
        args.output,
        embeddings=embeddings,
        episode_ids=[item.episode_id for item in decisions],
        decision_indices=[item.decision_index for item in decisions],
        extra=extra,
    )
    print(
        f"Cached {len(decisions)} VLM prefix embeddings "
        f"(hidden_size={embeddings.shape[1]}, dtype={dtype}) -> {args.output}"
    )


if __name__ == "__main__":
    main()
