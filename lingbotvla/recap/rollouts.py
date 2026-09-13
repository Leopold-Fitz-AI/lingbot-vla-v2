"""Read and validate raw RECAP rollout bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RolloutDecision:
    episode_id: str
    task: str
    task_name: str
    decision_index: int
    observation_path: Path
    reward: float
    empirical_return: float
    executed_action_length: int
    terminated: bool
    truncated: bool
    intervention: bool


def _decision_reward(
    *,
    executed_action_length: int,
    is_final: bool,
    success: bool,
    failure_penalty: float,
    gamma: float,
) -> float:
    """Aggregate action-level time-to-success rewards into one decision."""

    if executed_action_length < 0:
        raise ValueError("executed_action_length must be non-negative")
    elapsed_steps = executed_action_length
    if is_final:
        elapsed_steps = max(executed_action_length - 1, 0)
    if gamma == 1.0:
        time_cost = -float(elapsed_steps)
    else:
        time_cost = -(1.0 - gamma**elapsed_steps) / (1.0 - gamma)
    if not is_final or success:
        return time_cost
    return time_cost - (gamma**elapsed_steps) * float(failure_penalty)


def load_rollout_decisions(
    rollout_dir: str | Path,
    *,
    failure_penalty: float = 1000.0,
    gamma: float = 1.0,
) -> list[RolloutDecision]:
    """Load manifests and compute duration-aware Monte-Carlo returns."""

    if failure_penalty <= 0:
        raise ValueError("failure_penalty must be positive")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    root = Path(rollout_dir).expanduser().resolve()
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No RECAP manifest.json files found under {root}")

    decisions: list[RolloutDecision] = []
    seen_episode_ids: set[str] = set()
    for manifest_path in manifests:
        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        episode_id = str(manifest.get("episode_id", ""))
        if not episode_id:
            raise ValueError(f"Missing episode_id in {manifest_path}")
        if episode_id in seen_episode_ids:
            raise ValueError(f"Duplicate episode_id {episode_id!r}")
        seen_episode_ids.add(episode_id)

        steps = manifest.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"Episode {episode_id!r} has no recorded decisions")
        success = bool(manifest.get("success", False))
        task = str(manifest.get("task", ""))
        metadata = manifest.get("metadata") or {}
        task_name = str(metadata.get("task_name") or task)

        explicit_rewards = [step.get("reward") is not None for step in steps]
        if any(explicit_rewards) and not all(explicit_rewards):
            raise ValueError(f"Episode {episode_id!r} mixes explicit and implicit rewards")

        rewards: list[float] = []
        durations: list[int] = []
        for index, step in enumerate(steps):
            duration = int(step.get("executed_action_length", 0))
            durations.append(duration)
            if all(explicit_rewards):
                rewards.append(float(step["reward"]))
            else:
                rewards.append(
                    _decision_reward(
                        executed_action_length=duration,
                        is_final=index == len(steps) - 1,
                        success=success,
                        failure_penalty=failure_penalty,
                        gamma=gamma,
                    )
                )

        returns = [0.0] * len(steps)
        running = 0.0
        for index in range(len(steps) - 1, -1, -1):
            running = rewards[index] + (gamma ** durations[index]) * running
            returns[index] = running

        for index, (step, reward, empirical_return, duration) in enumerate(zip(steps, rewards, returns, durations)):
            observation_path = manifest_path.parent / str(step["file"])
            if not observation_path.is_file():
                raise FileNotFoundError(f"Missing decision data for episode {episode_id!r}: {observation_path}")
            decisions.append(
                RolloutDecision(
                    episode_id=episode_id,
                    task=task,
                    task_name=task_name,
                    decision_index=int(step.get("decision_index", index)),
                    observation_path=observation_path,
                    reward=reward,
                    empirical_return=empirical_return,
                    executed_action_length=duration,
                    terminated=bool(step.get("terminated", False)),
                    truncated=bool(step.get("truncated", False)),
                    intervention=bool(step.get("intervention", False)),
                )
            )
    return decisions


def load_decision_state(
    decision: RolloutDecision,
    *,
    state_key: str = "observation.state",
) -> np.ndarray:
    """Load and flatten one raw state array from a decision bundle."""

    archive_key = f"observation::{state_key}"
    with np.load(decision.observation_path, allow_pickle=False) as archive:
        if archive_key not in archive.files:
            raise KeyError(
                f"State key {state_key!r} is absent from {decision.observation_path}; "
                f"available arrays: {archive.files}"
            )
        state = np.asarray(archive[archive_key], dtype=np.float32).reshape(-1)
    if state.size == 0 or not np.isfinite(state).all():
        raise ValueError(f"Invalid state in {decision.observation_path}")
    return state


def _is_image_key(name: str) -> bool:
    lowered = str(name).lower()
    return "image" in lowered or "rgb" in lowered


def _hwc_or_chw_to_chw(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"Image array must be HWC or CHW, got shape {array.shape}")
    height, width, last = array.shape
    first = array.shape[0]
    channel_first = first in (1, 3, 4) and first < min(array.shape[1], array.shape[2])
    if channel_first:
        return np.asarray(array, dtype=np.float32)
    return np.transpose(array, (2, 0, 1)).astype(np.float32)


def _resize_chw(image: np.ndarray, image_size: int) -> np.ndarray:
    _, height, width = image.shape
    if height == image_size and width == image_size:
        return image
    row_index = (np.arange(image_size) * height / image_size).astype(np.int64)
    col_index = (np.arange(image_size) * width / image_size).astype(np.int64)
    return image[:, row_index][:, :, col_index]


def load_decision_images(
    decision: RolloutDecision,
    *,
    image_keys: list[str] | tuple[str, ...] | None = None,
    image_size: int | None = 64,
) -> np.ndarray:
    """Load camera images as ``[V, C, H, W]`` float32 in ``[0, 1]``.

    Keys are taken from the NPZ ``observation::`` arrays. By default every key
    whose name contains ``image`` or ``rgb`` is used, sorted for stability.
    """

    with np.load(decision.observation_path, allow_pickle=False) as archive:
        available = [name.removeprefix("observation::") for name in archive.files if name.startswith("observation::")]
        if image_keys is None:
            selected = [name for name in available if _is_image_key(name)]
        else:
            selected = [str(name) for name in image_keys]
        if not selected:
            raise KeyError(
                f"No camera images in {decision.observation_path}; available arrays: {archive.files}"
            )
        selected = sorted(selected)
        views = []
        for key in selected:
            archive_key = f"observation::{key}"
            if archive_key not in archive.files:
                raise KeyError(f"Image key {key!r} is absent from {decision.observation_path}")
            image = _hwc_or_chw_to_chw(np.asarray(archive[archive_key]))
            if image_size is not None:
                if int(image_size) <= 0:
                    raise ValueError(f"image_size must be positive, got {image_size}")
                image = _resize_chw(image, int(image_size))
            if image.max() > 1.0:
                image = image / 255.0
            image = np.clip(image, 0.0, 1.0)
            if not np.isfinite(image).all():
                raise ValueError(f"Non-finite image for {key!r} in {decision.observation_path}")
            views.append(image)
    stacked = np.stack(views, axis=0)
    channel_counts = {view.shape[0] for view in views}
    if len(channel_counts) != 1:
        raise ValueError(f"Mixed image channel counts in {decision.observation_path}: {sorted(channel_counts)}")
    return stacked.astype(np.float32, copy=False)


def group_decisions_by_episode(
    decisions: list[RolloutDecision],
) -> dict[str, list[RolloutDecision]]:
    """Group decisions by episode, preserving decision-index order."""

    grouped: dict[str, list[RolloutDecision]] = {}
    for decision in decisions:
        grouped.setdefault(decision.episode_id, []).append(decision)
    for episode_id, items in grouped.items():
        grouped[episode_id] = sorted(items, key=lambda item: item.decision_index)
    return grouped


def decisions_to_json_episode(
    decisions: list[RolloutDecision],
    values: list[float] | np.ndarray,
) -> dict[str, Any]:
    """Build the JSON episode consumed by ``recap_label_rollouts.py``."""

    if not decisions:
        raise ValueError("decisions must not be empty")
    if len(decisions) != len(values):
        raise ValueError("decisions and values must have the same length")
    episode_ids = {decision.episode_id for decision in decisions}
    if len(episode_ids) != 1:
        raise ValueError("All decisions must belong to one episode")
    steps = []
    for decision, value in zip(decisions, values):
        steps.append(
            {
                "decision_index": decision.decision_index,
                "observation_ref": str(decision.observation_path),
                "value": float(value),
                "reward": decision.reward,
                "terminated": decision.terminated,
                "truncated": decision.truncated,
                "valid": decision.executed_action_length > 0,
                "executed_action_length": decision.executed_action_length,
                "intervention": decision.intervention,
            }
        )
    return {
        "episode_id": decisions[0].episode_id,
        "task": decisions[0].task,
        "task_name": decisions[0].task_name,
        "success": bool(decisions[-1].terminated and not decisions[-1].truncated),
        "steps": steps,
    }
