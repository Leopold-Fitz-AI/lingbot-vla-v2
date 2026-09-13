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
    n_step_advantages,
    positive_quantile_threshold,
    time_to_success_rewards,
)
from .value import (
    CategoricalValueHead,
    FrozenVLMValueModel,
    StateTaskValueModel,
    VisualTaskValueModel,
    VLMPooledValueModel,
    categorical_value_loss,
    discretize_returns,
    masked_mean_pool,
    monte_carlo_returns,
    n_step_returns,
)


__all__ = [
    "CategoricalValueHead",
    "FrozenVLMValueModel",
    "RecapAdvantageLabel",
    "RecapCondition",
    "StateTaskValueModel",
    "VLMPooledValueModel",
    "VisualTaskValueModel",
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
    "masked_mean_pool",
    "load_recap_adapter_registry",
    "maybe_drop_recap_condition",
    "monte_carlo_returns",
    "n_step_advantages",
    "n_step_returns",
    "normalize_recap_condition",
    "positive_quantile_threshold",
    "recap_condition_id",
    "time_to_success_rewards",
]
