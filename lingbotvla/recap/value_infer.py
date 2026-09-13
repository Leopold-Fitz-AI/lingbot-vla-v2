"""Load RECAP value checkpoints and score recorded rollout decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from lingbotvla.recap.rollouts import (
    RolloutDecision,
    load_decision_images,
    load_decision_state,
)
from lingbotvla.recap.value import (
    StateTaskValueModel,
    VisualTaskValueModel,
    VLMPooledValueModel,
)
from lingbotvla.recap.vlm_pool import load_vlm_embedding_cache, match_rollout_embeddings


def load_value_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "model_type" not in payload:
        raise ValueError(f"Invalid RECAP value checkpoint: {checkpoint_path}")
    return payload


def build_value_model(checkpoint: dict[str, Any]):
    model_type = checkpoint.get("model_type")
    config = checkpoint["model_config"]
    if model_type == "state_task_categorical_value":
        model = StateTaskValueModel(**config)
    elif model_type == "visual_task_categorical_value":
        model = VisualTaskValueModel(**config)
    elif model_type == "vlm_pooled_categorical_value":
        model = VLMPooledValueModel(**config)
    else:
        raise ValueError(f"Unsupported value checkpoint type: {model_type!r}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, model_type


def decision_feature_tensor(
    decisions: Sequence[RolloutDecision],
    checkpoint: dict[str, Any],
    *,
    embedding_cache: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(features, task_ids)`` aligned with ``decisions``."""

    task_to_id = checkpoint["task_to_id"]
    unknown_tasks = sorted({item.task_name for item in decisions if item.task_name not in task_to_id})
    if unknown_tasks:
        raise ValueError("Value checkpoint does not contain these tasks: " + ", ".join(unknown_tasks))
    task_tensor = torch.tensor([task_to_id[item.task_name] for item in decisions], dtype=torch.long)
    model_type = checkpoint["model_type"]
    if model_type == "visual_task_categorical_value":
        image_keys = checkpoint.get("image_keys")
        image_size = int(checkpoint.get("image_size") or checkpoint["model_config"]["image_size"])
        features = np.stack(
            [
                load_decision_images(item, image_keys=image_keys, image_size=image_size)
                for item in decisions
            ]
        )
        feature_tensor = torch.from_numpy(features).float()
    elif model_type == "vlm_pooled_categorical_value":
        cache_path = embedding_cache or checkpoint.get("embedding_cache")
        if not cache_path:
            raise ValueError("vlm_pooled checkpoints require --embedding-cache or a stored cache path")
        feature_tensor = match_rollout_embeddings(load_vlm_embedding_cache(cache_path), decisions)
    else:
        state_key = str(checkpoint["state_key"])
        states = np.stack([load_decision_state(item, state_key=state_key) for item in decisions])
        feature_tensor = torch.from_numpy(states).float()
        feature_tensor = (feature_tensor - checkpoint["state_mean"]) / checkpoint["state_std"]
    return feature_tensor, task_tensor


def predict_decision_values(
    decisions: Sequence[RolloutDecision],
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 512,
    embedding_cache: str | Path | None = None,
) -> list[float]:
    """Score each decision with a trained categorical value checkpoint."""

    if not decisions:
        return []
    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    checkpoint = load_value_checkpoint(checkpoint_path)
    model, _model_type = build_value_model(checkpoint)
    torch_device = torch.device(device)
    model.to(torch_device).eval()
    feature_tensor, task_tensor = decision_feature_tensor(
        decisions,
        checkpoint,
        embedding_cache=embedding_cache,
    )
    predictions: list[float] = []
    with torch.no_grad():
        for start in range(0, len(decisions), int(batch_size)):
            stop = start + int(batch_size)
            features = feature_tensor[start:stop].to(torch_device)
            if getattr(model, "uses_task_id", True):
                prediction = model.expected_value(features, task_tensor[start:stop].to(torch_device))
            else:
                prediction = model.expected_value(features)
            predictions.extend(prediction.cpu().tolist())
    return predictions
