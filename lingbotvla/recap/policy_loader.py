"""Load a frozen LingBot policy checkpoint for offline prefix encoding.

This is the production weight-loading path for
``scripts/recap_cache_vlm_embeddings.py``. It mirrors
``deploy/lingbot_vla_v2_policy.py`` ``load_vla`` without importing the
deployment server (which seeds global RNG state at import time). Heavy
dependencies (transformers, the VLA model) are imported lazily so lightweight
unit tests can monkeypatch this loader without a GPU or a checkpoint.
"""

from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml


_QWEN_TEXT_CONFIG_KEYS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "rms_norm_eps",
    "rope_theta",
    "vocab_size",
    "max_position_embeddings",
    "hidden_act",
    "tie_word_embeddings",
    "tokenizer_path",
)


def _merge_qwen_config(config: Any, qwen_config: Any) -> None:
    """Copy the Qwen3-VL backbone fields onto the VLA config (deploy parity)."""

    config_dict = qwen_config.to_dict() if hasattr(qwen_config, "to_dict") else qwen_config
    text_config = config_dict.get("text_config", {})
    for key in _QWEN_TEXT_CONFIG_KEYS:
        if key in text_config:
            setattr(config, key, text_config[key])
        elif key in config_dict:
            setattr(config, key, config_dict[key])
    if "vision_config" in config_dict:
        config.vision_config = qwen_config.vision_config


def training_config_path_for_checkpoint(checkpoint_path: str | Path) -> Path:
    """Locate ``lingbotvla_cli.yaml`` three levels above the checkpoint.

    Matches ``deploy/lingbot_vla_v2_policy.py``: the path is NOT resolved, so
    a checkpoint directory reached through a symlink still finds the training
    config next to the run directory the caller passed.
    """

    return Path(checkpoint_path).expanduser().parent.parent.parent / "lingbotvla_cli.yaml"


def load_frozen_flow_model(
    checkpoint_path: str | Path,
    robot_config: str | Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    norm_stats_path: str | Path | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a frozen policy and return ``(flow_model, feature_transform, meta)``.

    ``flow_model`` is the ``FlowMatchingV2`` module (``policy.model``) moved to
    ``device``/``dtype`` and put in eval mode; it satisfies the interface of
    :func:`lingbotvla.recap.vlm_pool.encode_policy_prefix_hidden`.
    """

    from safetensors import safe_open
    from transformers import AutoConfig

    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.models import build_processor
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
        LingbotVLAV2Config,
    )
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import (
        LingbotVlaV2Policy,
    )
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import (
        apply_lingbot_qwen3_vl_patch,
    )

    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Policy checkpoint directory does not exist: {checkpoint_path}")

    training_config_path = training_config_path_for_checkpoint(checkpoint_path)
    if not training_config_path.is_file():
        raise FileNotFoundError(
            f"Training config not found at {training_config_path}; the checkpoint "
            "must live under <run>/.../<checkpoint> next to lingbotvla_cli.yaml"
        )
    with training_config_path.open("r") as file:
        training_config = yaml.safe_load(file)

    training_model_config = dict(training_config["model"])
    training_model_config.update(training_config["train"])
    config = LingbotVLAV2Config(**training_model_config)
    for key, value in training_model_config.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    config.attention_implementation = "eager"

    training_base_model = training_config["model"]["tokenizer_path"]
    if "qwen3" not in str(training_base_model).lower() or "vl" not in str(training_base_model).lower():
        raise ValueError(f"Unsupported base model of {checkpoint_path}: {training_base_model}")
    base_model_path = os.environ.get("QWEN3VL_PATH", training_base_model)
    config.tokenizer_path = base_model_path

    qwen_config = AutoConfig.from_pretrained(base_model_path)
    _merge_qwen_config(config, qwen_config)
    if "vocab_size" in training_config["model"] and training_config["model"]["vocab_size"] != 0:
        config.vocab_size = training_config["model"]["vocab_size"]
    config.use_cache = True

    processor = build_processor(base_model_path)
    data_config = SimpleNamespace(**training_config["data"])

    apply_lingbot_qwen3_vl_patch()
    policy = LingbotVlaV2Policy(config, eval=True)

    safetensors_files = sorted(glob(str(checkpoint_path / "*.safetensors")))
    if not safetensors_files:
        raise FileNotFoundError(f"No *.safetensors files found in {checkpoint_path}")
    merged_weights: dict[str, torch.Tensor] = {}
    for file_path in safetensors_files:
        with safe_open(file_path, framework="pt", device="cpu") as archive:
            for key in archive.keys():
                merged_weights[key] = archive.get_tensor(key)
    policy.load_state_dict(merged_weights, strict=True)

    policy = policy.to(device=device, dtype=dtype).eval()
    flow_model = policy.model

    resolved_norm_stats = norm_stats_path or getattr(data_config, "norm_stats_file", None)
    feature_transform = FeatureTransform(
        str(robot_config),
        data_config,
        config,
        processor,
        chunk_size=config.chunk_size,
        norm_stats_path=resolved_norm_stats,
    )

    meta = {
        "policy_checkpoint": str(checkpoint_path),
        "robot_config": str(robot_config),
        "base_model_path": str(base_model_path),
        "dtype": str(dtype),
        "hidden_size": int(config.hidden_size),
        "image_size": int(getattr(data_config, "img_size", 256)),
    }
    return flow_model, feature_transform, meta
