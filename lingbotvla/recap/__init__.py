"""Utilities for RECAP-style policy conditioning and value learning."""

from .adapter import (
    apply_recap_velocity_lora,
    build_recap_condition_embedding,
    load_counterfactual_decision_map,
    load_recap_adapter_registry,
)
from .cfg import (
    combine_cfg_velocities,
    combine_cfg_velocity_batch,
    duplicate_cfg_denoise_inputs,
)
from .conditioning import (
    RecapCondition,
    format_recap_prompt,
    maybe_drop_recap_condition,
    normalize_recap_condition,
    recap_condition_id,
)
from .labels import (
    RecapAdvantageLabel,
    compute_advantages,
    label_advantages,
    time_to_success_rewards,
)
from .value import (
    CategoricalValueHead,
    StateTaskValueModel,
    categorical_value_loss,
    discretize_returns,
    monte_carlo_returns,
)


__all__ = [
    "CategoricalValueHead",
    "RecapAdvantageLabel",
    "RecapCondition",
    "StateTaskValueModel",
    "apply_recap_velocity_lora",
    "build_recap_condition_embedding",
    "categorical_value_loss",
    "combine_cfg_velocities",
    "combine_cfg_velocity_batch",
    "compute_advantages",
    "duplicate_cfg_denoise_inputs",
    "discretize_returns",
    "format_recap_prompt",
    "label_advantages",
    "load_counterfactual_decision_map",
    "load_recap_adapter_registry",
    "maybe_drop_recap_condition",
    "monte_carlo_returns",
    "normalize_recap_condition",
    "recap_condition_id",
    "time_to_success_rewards",
]
