"""Turn recorded RECAP rollout decisions into policy prefix inputs.

This mirrors the deployment preprocessing in ``deploy/lingbot_vla_v2_policy.py``
(``resize_image`` + ``FeatureTransform.apply(policy_eval=True)``) without
importing the deployment server. The value encoder never sees an advantage
condition: the RECAP indicator is pinned to the null label (-1) so the prompt
is the raw task text.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torchvision.transforms.v2 import Resize

from .rollouts import RolloutDecision


def _decision_context(decision: RolloutDecision) -> str:
    return f"{decision.episode_id}::{decision.decision_index}"


def load_decision_observation(
    decision: RolloutDecision,
    image_keys: Sequence[str],
    state_key: str = "observation.state",
) -> dict[str, Any]:
    """Rebuild the raw deployment observation dict from a decision NPZ.

    Images are kept exactly as recorded (HWC uint8); the state stays a flat
    vector. ``"task"`` is injected from the rollout manifest. Missing keys
    raise fail-closed errors naming ``episode_id::decision_index``.
    """

    context = _decision_context(decision)
    image_keys = [str(key) for key in image_keys]
    if not image_keys:
        raise ValueError(f"Decision {context}: at least one image key is required")

    observation: dict[str, Any] = {}
    with np.load(decision.observation_path, allow_pickle=False) as archive:
        available = [name.removeprefix("observation::") for name in archive.files if name.startswith("observation::")]
        for key in image_keys:
            archive_key = f"observation::{key}"
            if archive_key not in archive.files:
                raise KeyError(
                    f"Decision {context}: image key {key!r} is absent from "
                    f"{decision.observation_path}; available arrays: {sorted(available)}"
                )
            observation[key] = np.asarray(archive[archive_key])
        state_archive_key = f"observation::{state_key}"
        if state_archive_key not in archive.files:
            raise KeyError(
                f"Decision {context}: state key {state_key!r} is absent from "
                f"{decision.observation_path}; available arrays: {sorted(available)}"
            )
        state = np.asarray(archive[state_archive_key], dtype=np.float32).reshape(-1)
    if state.size == 0 or not np.isfinite(state).all():
        raise ValueError(f"Decision {context}: invalid state in {decision.observation_path}")
    observation[state_key] = state
    observation["task"] = str(decision.task)
    return observation


def decision_to_prefix_inputs(
    observation: dict[str, Any],
    feature_transform: Any,
    image_size: int,
) -> dict[str, torch.Tensor]:
    """Replicate deployment preprocessing for one decision.

    Resizes every configured camera to ``image_size`` (deployment
    ``resize_image``), pins the RECAP indicator to the null condition, converts
    NumPy arrays to tensors, and runs ``feature_transform.apply(item,
    policy_eval=True)``. Returns the single-sample tensors consumed by
    :func:`lingbotvla.recap.vlm_pool.encode_policy_prefix_hidden`; the caller
    stacks the batch dimension.
    """

    if int(image_size) <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    item = dict(observation)
    if getattr(feature_transform, "recap_enabled", False):
        # The value function must not see an advantage condition: null (-1)
        # keeps the prompt as the raw task text with no "Advantage:" prefix.
        item[feature_transform.recap_indicator_key] = -1

    org_features = getattr(feature_transform, "org_features", None) or {}
    image_features = list(org_features.get("images", []))
    if not image_features:
        image_features = sorted(key for key in item if ".images." in str(key))
    if not image_features:
        raise ValueError("No camera image features configured for prefix encoding")

    resize = Resize((int(image_size), int(image_size)))
    for image_key in image_features:
        if image_key not in item:
            raise ValueError(f"Missing camera key {image_key!r} in decision observation")
        image = item[image_key]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(
                f"Camera {image_key!r} must be HWC with 3 channels, got shape {array.shape}"
            )
        tensor = torch.as_tensor(array).permute(2, 0, 1).contiguous()
        item[image_key] = resize(tensor.to(dtype=torch.float32))

    for key, value in list(item.items()):
        if isinstance(value, np.ndarray):
            item[key] = torch.from_numpy(value)

    transformed = feature_transform.apply(item, policy_eval=True)
    required = ("images", "img_masks", "state", "lang_tokens", "lang_masks", "image_grid_thw")
    missing = [key for key in required if transformed.get(key) is None]
    if missing:
        raise ValueError(
            f"feature_transform.apply(policy_eval=True) did not produce {missing}; "
            "the prefix encoder requires image_grid_thw from the Qwen3-VL image processor"
        )
    result = {key: transformed[key] for key in required}
    # unapply() needs the joint masks to slice padded state/action tensors;
    # pass them through when the transform produces them.
    for key in ("state_joint_mask", "action_joint_mask"):
        if transformed.get(key) is not None:
            result[key] = transformed[key]
    return result
