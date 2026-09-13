"""Frozen VLM prefix encoding and embedding-cache helpers for RECAP value."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor


CACHE_SCHEMA_VERSION = 1

# ``QwenvlWithExpertV2Model.get_image_features`` caches precomputed grid
# metadata on the module when ``config.precompute_grid_thw`` is enabled and
# only recomputes while ``position_embeddings is None``. A later batch with a
# different image count would silently reuse the stale cache (and then crash
# on the mismatched split sizes), so prefix encoding resets it per call.
_VISUAL_PRECOMPUTE_ATTRS = (
    "pos_embeds",
    "position_embeddings",
    "cu_seqlens",
    "visual_split_sizes",
    "visual_max_seqlen",
)


def _reset_visual_precompute_cache(flow_model: Any) -> None:
    expert = getattr(flow_model, "qwenvl_with_expert", None)
    config = getattr(expert, "config", None)
    if expert is None or not getattr(config, "precompute_grid_thw", False):
        return
    for attr in _VISUAL_PRECOMPUTE_ATTRS:
        if hasattr(expert, attr):
            setattr(expert, attr, None)


def _prefix_attention_mask(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """Prefix attention used by LingBot ``embed_prefix`` / ``sample_actions``.

    Kept here so Recap value encoding does not import the VLA utils module
    (and its optional einops dependency) during unit tests.
    """

    if att_masks.ndim != 2 or pad_masks.ndim != 2:
        raise ValueError("prefix pad/att masks must be rank 2")
    cumsum = torch.cumsum(att_masks.to(dtype=torch.int32), dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def encode_policy_prefix_hidden(
    flow_model: Any,
    images: Tensor,
    img_masks: Tensor,
    lang_tokens: Tensor,
    lang_masks: Tensor,
    image_grid_thw: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run the frozen VLA prefix stack and return ``(hidden [B, T, D], pad_mask)``.

    ``flow_model`` is a LingBot ``FlowMatchingV2`` (or a test double) exposing
    ``embed_prefix`` and ``qwenvl_with_expert.forward``. The 6B weights stay
    outside the value checkpoint; this function is ``no_grad``.
    """

    _reset_visual_precompute_cache(flow_model)
    with torch.no_grad():
        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = flow_model.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            image_grid_thw=image_grid_thw,
        )
        prefix_att_2d_masks = _prefix_attention_mask(prefix_pad_masks, prefix_att_masks)
        outputs_embeds, _, _ = flow_model.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            vlm_position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=False,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        hidden = outputs_embeds[0]
    if hidden is None:
        raise ValueError("VLM prefix encoder returned no hidden states")
    if hidden.ndim != 3:
        raise ValueError(f"VLM prefix hidden states must be [B, T, D], got {tuple(hidden.shape)}")
    return hidden.detach(), prefix_pad_masks.detach()


def _cache_key(episode_id: str, decision_index: int) -> str:
    return f"{episode_id}::{int(decision_index)}"


def write_vlm_embedding_cache(
    path: str | Path,
    *,
    embeddings: Tensor,
    episode_ids: Sequence[str],
    decision_indices: Sequence[int],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomically write a fail-closed VLM embedding cache."""

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must have shape [N, D], got {tuple(embeddings.shape)}")
    if len(episode_ids) != embeddings.shape[0] or len(decision_indices) != embeddings.shape[0]:
        raise ValueError("episode_ids, decision_indices, and embeddings must be aligned")
    if not torch.isfinite(embeddings).all():
        raise ValueError("VLM embeddings contain NaN or infinity")
    keys = [_cache_key(str(episode), int(index)) for episode, index in zip(episode_ids, decision_indices)]
    if len(set(keys)) != len(keys):
        raise ValueError("VLM embedding cache has duplicate episode/decision keys")
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_type": "vlm_pooled_embeddings",
        "pool": "masked_mean",
        "hidden_size": int(embeddings.shape[1]),
        "episode_ids": [str(item) for item in episode_ids],
        "decision_indices": [int(item) for item in decision_indices],
        "embeddings": embeddings.detach().cpu().float(),
    }
    if extra:
        overlap = set(extra) & set(payload)
        if overlap:
            raise ValueError(f"Cache extra keys collide with reserved fields: {sorted(overlap)}")
        payload.update(extra)
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def load_vlm_embedding_cache(path: str | Path) -> dict[str, Any]:
    cache_path = Path(path).expanduser().resolve()
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported VLM embedding cache schema in {cache_path}")
    embeddings = payload.get("embeddings")
    if not torch.is_tensor(embeddings) or embeddings.ndim != 2:
        raise ValueError("VLM embedding cache must contain a [N, D] embeddings tensor")
    episode_ids = payload.get("episode_ids")
    decision_indices = payload.get("decision_indices")
    if not isinstance(episode_ids, list) or not isinstance(decision_indices, list):
        raise ValueError("VLM embedding cache must contain episode_ids and decision_indices lists")
    if len(episode_ids) != embeddings.shape[0] or len(decision_indices) != embeddings.shape[0]:
        raise ValueError("VLM embedding cache keys are not aligned with embeddings")
    hidden_size = payload.get("hidden_size", embeddings.shape[1])
    if int(hidden_size) != int(embeddings.shape[1]):
        raise ValueError(
            f"Cache hidden_size {hidden_size} does not match embeddings {tuple(embeddings.shape)}"
        )
    payload["embeddings"] = embeddings.float()
    payload["hidden_size"] = int(hidden_size)
    return payload


def match_rollout_embeddings(cache: dict[str, Any], decisions: Sequence[Any]) -> Tensor:
    """Return embeddings in rollout-decision order. Missing keys raise KeyError."""

    index = {
        _cache_key(str(episode), int(decision)): row
        for row, (episode, decision) in enumerate(
            zip(cache["episode_ids"], cache["decision_indices"])
        )
    }
    rows = []
    missing = []
    for decision in decisions:
        key = _cache_key(str(decision.episode_id), int(decision.decision_index))
        if key not in index:
            missing.append(key)
            continue
        rows.append(index[key])
    if missing:
        raise KeyError("Missing VLM embeddings for: " + ", ".join(missing))
    return cache["embeddings"][rows]
