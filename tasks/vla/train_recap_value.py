#!/usr/bin/env python3
"""Train a RECAP distributional value baseline from recorded rollouts.

``--encoder state_task`` (default) is the proprio + task-id MLP.
``--encoder visual_task`` encodes rollout cameras with a small conv net.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lingbotvla.recap.rollouts import (  # noqa: E402
    load_decision_images,
    load_decision_state,
    load_rollout_decisions,
)
from lingbotvla.recap.value import (  # noqa: E402
    StateTaskValueModel,
    VisualTaskValueModel,
    VLMPooledValueModel,
    categorical_value_loss,
)
from lingbotvla.recap.vlm_pool import load_vlm_embedding_cache, match_rollout_embeddings  # noqa: E402


def parse_args(
    argv: list[str] | None = None,
    *,
    encoder_default: str = "state_task",
    lock_encoder: bool = False,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--output", required=True, help="Value checkpoint path (.pt)")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--failure-penalty", type=float, default=1000.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--value-min", type=float, default=-2000.0)
    parser.add_argument("--value-max", type=float, default=0.0)
    parser.add_argument("--num-bins", type=int, default=201)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--task-embedding-dim", type=int, default=64)
    if lock_encoder:
        parser.set_defaults(encoder=encoder_default)
    else:
        parser.add_argument(
            "--encoder",
            choices=("state_task", "visual_task", "vlm_pooled"),
            default=encoder_default,
            help=(
                "state_task is the proprio baseline; visual_task encodes rollout cameras; "
                "vlm_pooled trains a categorical head on frozen VLM embeddings."
            ),
        )
    parser.add_argument(
        "--embedding-cache",
        default=None,
        help="Required for --encoder vlm_pooled. Cache from recap_cache_vlm_embeddings.py.",
    )
    parser.add_argument(
        "--projector-hidden-size",
        type=int,
        default=None,
        help="Optional MLP width on top of pooled VLM embeddings. Default: linear head only.",
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--image-keys",
        nargs="*",
        default=None,
        help="Optional explicit observation image keys. Default: every key containing 'image' or 'rgb'.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--selection-metric",
        choices=("loss", "mae"),
        default="loss",
        help="Metric used to select the persisted best checkpoint.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--stratify-success",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Split whole episodes separately within successful and failed strata.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    return parser.parse_args(argv)


def _select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _split_episodes(
    episode_ids: list[str],
    *,
    validation_fraction: float,
    seed: int,
    episode_strata: dict[str, bool] | None = None,
) -> tuple[set[str], set[str]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    unique = sorted(set(episode_ids))
    if validation_fraction == 0.0 or len(unique) == 1:
        return set(unique), set()

    rng = random.Random(seed)
    if episode_strata is None:
        rng.shuffle(unique)
        validation_count = max(1, round(len(unique) * validation_fraction))
        validation_count = min(validation_count, len(unique) - 1)
        validation = set(unique[:validation_count])
        return set(unique[validation_count:]), validation

    missing = sorted(set(unique) - set(episode_strata))
    if missing:
        raise ValueError(f"Missing episode strata for: {missing}")
    grouped: dict[bool, list[str]] = {}
    for episode_id in unique:
        grouped.setdefault(bool(episode_strata[episode_id]), []).append(episode_id)

    validation: set[str] = set()
    for stratum in sorted(grouped):
        episodes = grouped[stratum]
        rng.shuffle(episodes)
        if len(episodes) <= 1:
            continue
        validation_count = max(1, round(len(episodes) * validation_fraction))
        validation_count = min(validation_count, len(episodes) - 1)
        validation.update(episodes[:validation_count])
    if not validation:
        shuffled = list(unique)
        rng.shuffle(shuffled)
        validation.add(shuffled[0])
    return set(unique) - validation, validation


def _model_logits(model, features, task_id):
    if getattr(model, "uses_task_id", True):
        return model(features, task_id)
    return model(features)


def _evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    losses = []
    absolute_errors = []
    with torch.no_grad():
        for batch in loader:
            features, task_id, returns = batch
            features = features.to(device)
            task_id = task_id.to(device)
            returns = returns.to(device)
            logits = _model_logits(model, features, task_id)
            loss = categorical_value_loss(logits, returns, model.value_support)
            prediction = model.value_head.expected_value_from_logits(logits)
            losses.append(float(loss.item()) * features.shape[0])
            absolute_errors.append(float((prediction - returns).abs().sum().item()))
    denominator = max(len(loader.dataset), 1)
    return sum(losses) / denominator, sum(absolute_errors) / denominator


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    decisions = load_rollout_decisions(
        args.rollout_dir,
        failure_penalty=args.failure_penalty,
        gamma=args.gamma,
    )
    task_names = sorted({decision.task_name for decision in decisions})
    task_to_id = {task: index for index, task in enumerate(task_names)}
    task_tensor = torch.tensor([task_to_id[item.task_name] for item in decisions], dtype=torch.long)
    return_tensor = torch.tensor([item.empirical_return for item in decisions], dtype=torch.float32)
    image_keys = list(args.image_keys) if args.image_keys else None
    state_mean = None
    state_std = None
    state_dim = None
    in_channels = None
    vlm_hidden_size = None
    if args.encoder == "vlm_pooled":
        if not args.embedding_cache:
            raise ValueError("--encoder vlm_pooled requires --embedding-cache")
        cache = load_vlm_embedding_cache(args.embedding_cache)
        feature_tensor = match_rollout_embeddings(cache, decisions)
        vlm_hidden_size = int(cache["hidden_size"])
    elif args.encoder == "visual_task":
        images = [
            load_decision_images(decision, image_keys=image_keys, image_size=args.image_size)
            for decision in decisions
        ]
        shapes = {tuple(image.shape) for image in images}
        if len(shapes) != 1:
            raise ValueError(f"All visual observations must share shape, got {sorted(shapes)}")
        feature_tensor = torch.from_numpy(np.stack(images)).float()
        in_channels = int(feature_tensor.shape[2])
    else:
        states = [load_decision_state(decision, state_key=args.state_key) for decision in decisions]
        state_dims = {state.shape[0] for state in states}
        if len(state_dims) != 1:
            raise ValueError(f"All states must have the same flattened dimension, got {sorted(state_dims)}")
        state_dim = state_dims.pop()
        feature_tensor = torch.from_numpy(np.stack(states)).float()
    episode_success = {
        item.episode_id: bool(item.terminated and not item.truncated)
        for item in decisions
        if item.terminated or item.truncated
    }
    all_episode_ids = {item.episode_id for item in decisions}
    if set(episode_success) != all_episode_ids:
        missing = sorted(all_episode_ids - set(episode_success))
        raise ValueError(f"Episodes are missing a terminated/truncated final decision: {missing}")

    train_episodes, validation_episodes = _split_episodes(
        [item.episode_id for item in decisions],
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        episode_strata=episode_success if args.stratify_success else None,
    )
    train_indices = torch.tensor(
        [index for index, item in enumerate(decisions) if item.episode_id in train_episodes],
        dtype=torch.long,
    )
    validation_indices = torch.tensor(
        [index for index, item in enumerate(decisions) if item.episode_id in validation_episodes],
        dtype=torch.long,
    )
    if train_indices.numel() == 0:
        raise ValueError("Episode split produced no training decisions")

    if args.encoder == "state_task":
        state_mean = feature_tensor[train_indices].mean(dim=0)
        state_std = feature_tensor[train_indices].std(dim=0, unbiased=False).clamp(min=1e-6)
        feature_tensor = (feature_tensor - state_mean) / state_std

    train_dataset = TensorDataset(
        feature_tensor[train_indices],
        task_tensor[train_indices],
        return_tensor[train_indices],
    )
    validation_dataset = TensorDataset(
        feature_tensor[validation_indices],
        task_tensor[validation_indices],
        return_tensor[validation_indices],
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = _select_device(args.device)
    if args.encoder == "visual_task":
        model_type = "visual_task_categorical_value"
        model_config = {
            "num_tasks": len(task_to_id),
            "in_channels": in_channels,
            "image_size": args.image_size,
            "hidden_size": args.hidden_size,
            "task_embedding_dim": args.task_embedding_dim,
            "num_bins": args.num_bins,
            "value_min": args.value_min,
            "value_max": args.value_max,
        }
        model = VisualTaskValueModel(**model_config).to(device)
    elif args.encoder == "vlm_pooled":
        model_type = "vlm_pooled_categorical_value"
        model_config = {
            "hidden_size": vlm_hidden_size,
            "projector_hidden_size": args.projector_hidden_size,
            "num_bins": args.num_bins,
            "value_min": args.value_min,
            "value_max": args.value_max,
        }
        model = VLMPooledValueModel(**model_config).to(device)
    else:
        model_type = "state_task_categorical_value"
        model_config = {
            "state_dim": state_dim,
            "num_tasks": len(task_to_id),
            "hidden_size": args.hidden_size,
            "task_embedding_dim": args.task_embedding_dim,
            "num_bins": args.num_bins,
            "value_min": args.value_min,
            "value_max": args.value_max,
        }
        model = StateTaskValueModel(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_metric = float("inf")
    clipped_fraction = float(
        ((return_tensor < args.value_min) | (return_tensor > args.value_max)).float().mean().item()
    )
    split_summary = {
        "train_success_episodes": sum(episode_success[item] for item in train_episodes),
        "train_failure_episodes": sum(not episode_success[item] for item in train_episodes),
        "validation_success_episodes": sum(episode_success[item] for item in validation_episodes),
        "validation_failure_episodes": sum(not episode_success[item] for item in validation_episodes),
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, task_id, returns in train_loader:
            features = features.to(device)
            task_id = task_id.to(device)
            returns = returns.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = _model_logits(model, features, task_id)
            loss = categorical_value_loss(logits, returns, model.value_support)
            loss.backward()
            optimizer.step()

        train_loss, train_mae = _evaluate(model, train_loader, device)
        if len(validation_dataset):
            validation_loss, validation_mae = _evaluate(model, validation_loader, device)
            selection_metric = validation_mae if args.selection_metric == "mae" else validation_loss
        else:
            validation_loss, validation_mae = None, None
            selection_metric = train_mae if args.selection_metric == "mae" else train_loss
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "validation_loss": validation_loss,
            "validation_mae": validation_mae,
            "clipped_return_fraction": clipped_fraction,
        }
        print(json.dumps(metrics, sort_keys=True))

        if selection_metric < best_metric:
            best_metric = selection_metric
            checkpoint = {
                "schema_version": 1,
                "model_type": model_type,
                "model_config": model_config,
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "task_to_id": task_to_id,
                "state_key": args.state_key,
                "state_mean": state_mean,
                "state_std": state_std,
                "image_size": args.image_size if args.encoder == "visual_task" else None,
                "image_keys": image_keys,
                "embedding_cache": args.embedding_cache,
                "failure_penalty": args.failure_penalty,
                "gamma": args.gamma,
                "metrics": metrics,
                "train_episodes": sorted(train_episodes),
                "validation_episodes": sorted(validation_episodes),
                "episode_success": episode_success,
                "split_stratified_by_success": bool(args.stratify_success),
                "split_summary": split_summary,
                "selection_metric": args.selection_metric,
            }
            temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(output_path)

    print(
        json.dumps(
            {
                "checkpoint": str(output_path),
                "decisions": len(decisions),
                "episodes": len(train_episodes) + len(validation_episodes),
                "tasks": len(task_to_id),
                "device": str(device),
                "selection_metric": args.selection_metric,
                "best_selection_metric": best_metric,
                "split_summary": split_summary,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
