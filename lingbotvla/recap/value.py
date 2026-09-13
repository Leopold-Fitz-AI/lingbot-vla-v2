"""Distributional value-learning building blocks for RECAP.

The original RECAP recipe models Monte-Carlo returns with 201 categorical
value bins.  This module deliberately stays independent of a particular vision
encoder: callers can attach :class:`CategoricalValueHead` to any pooled VLM
representation while sharing the return and loss implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CategoricalValueHead(nn.Module):
    """Map context embeddings to a categorical value distribution."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_bins: int = 201,
        value_min: float = -2000.0,
        value_max: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_bins < 2:
            raise ValueError(f"num_bins must be at least 2, got {num_bins}")
        if value_max <= value_min:
            raise ValueError(f"value_max must be greater than value_min, got {value_min}..{value_max}")
        self.projection = nn.Linear(hidden_size, num_bins)
        self.register_buffer(
            "value_support",
            torch.linspace(float(value_min), float(value_max), num_bins),
            persistent=True,
        )

    @property
    def num_bins(self) -> int:
        return int(self.value_support.numel())

    def forward(self, context_embedding: Tensor) -> Tensor:
        return self.projection(context_embedding)

    def probabilities(self, context_embedding: Tensor) -> Tensor:
        return F.softmax(self(context_embedding), dim=-1)

    def expected_value_from_logits(self, logits: Tensor) -> Tensor:
        if logits.shape[-1] != self.num_bins:
            raise ValueError(f"Expected {self.num_bins} value logits, got shape {tuple(logits.shape)}")
        support = self.value_support.to(device=logits.device, dtype=logits.dtype)
        return (F.softmax(logits, dim=-1) * support).sum(dim=-1)

    def expected_value(self, context_embedding: Tensor) -> Tensor:
        return self.expected_value_from_logits(self(context_embedding))


def masked_mean_pool(hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
    """Pool a VLM sequence ``[B, T, D]`` to ``[B, D]``. 2D inputs pass through."""

    if hidden_states.ndim == 2:
        if attention_mask is not None:
            raise ValueError("attention_mask is only valid for sequence embeddings of shape [B, T, D]")
        return hidden_states
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must have shape [B, D] or [B, T, D], got {tuple(hidden_states.shape)}")
    if attention_mask is None:
        return hidden_states.mean(dim=1)
    mask = attention_mask.to(device=hidden_states.device)
    if mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"attention_mask shape {tuple(mask.shape)} must match hidden_states {tuple(hidden_states.shape[:2])}"
        )
    weights = mask.to(dtype=hidden_states.dtype)
    if not torch.isfinite(hidden_states).all():
        raise ValueError("hidden_states contains NaN or infinity")
    counts = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (hidden_states * weights.unsqueeze(-1)).sum(dim=1) / counts


class VLMPooledValueModel(nn.Module):
    """Categorical critic attached to a pooled VLM observation embedding.

    This is the original RECAP value interface ``p(V | o_t, ℓ)`` once a frozen
    VLM has already mapped images and language into one vector. The 6B policy
    is not a submodule: train this head on cached embeddings, or wrap it with
    :class:`FrozenVLMValueModel` for encode-on-the-fly.
    """

    uses_task_id = False

    def __init__(
        self,
        hidden_size: int,
        *,
        projector_hidden_size: int | None = None,
        num_bins: int = 201,
        value_min: float = -2000.0,
        value_max: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.hidden_size = int(hidden_size)
        if projector_hidden_size is None:
            self.projector = nn.Identity()
            head_size = self.hidden_size
        else:
            if int(projector_hidden_size) <= 0:
                raise ValueError(
                    f"projector_hidden_size must be positive, got {projector_hidden_size}"
                )
            head_size = int(projector_hidden_size)
            self.projector = nn.Sequential(
                nn.Linear(self.hidden_size, head_size),
                nn.LayerNorm(head_size),
                nn.SiLU(),
                nn.Linear(head_size, head_size),
                nn.SiLU(),
            )
        self.value_head = CategoricalValueHead(
            head_size,
            num_bins=num_bins,
            value_min=value_min,
            value_max=value_max,
        )

    @property
    def value_support(self) -> Tensor:
        return self.value_head.value_support

    def forward(self, embeddings: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        pooled = masked_mean_pool(embeddings, attention_mask)
        if pooled.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected pooled embedding size {self.hidden_size}, got {pooled.shape[-1]}"
            )
        return self.value_head(self.projector(pooled))

    def expected_value(self, embeddings: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.value_head.expected_value_from_logits(self(embeddings, attention_mask))


class FrozenVLMValueModel(nn.Module):
    """Trainable value head on a frozen observation encoder.

    ``encoder`` may be any module whose forward returns ``[B, D]``, ``[B, T, D]``,
    or ``(hidden, mask)``. It is frozen and excluded from
    :meth:`trainable_state_dict` so checkpoints stay head-sized.
    """

    uses_task_id = False

    def __init__(
        self,
        value_model: VLMPooledValueModel,
        *,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.value_model = value_model
        self.encoder = encoder
        if encoder is not None:
            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad = False

    @property
    def value_support(self) -> Tensor:
        return self.value_model.value_support

    def trainable_state_dict(self) -> dict[str, Tensor]:
        return self.value_model.state_dict()

    def forward(
        self,
        embeddings: Tensor | None = None,
        attention_mask: Tensor | None = None,
        *,
        encoder_inputs: tuple | None = None,
        **encoder_kwargs,
    ) -> Tensor:
        if embeddings is None:
            if self.encoder is None:
                raise ValueError("FrozenVLMValueModel requires embeddings or an encoder")
            with torch.no_grad():
                encoded = (
                    self.encoder(*encoder_inputs)
                    if encoder_inputs is not None
                    else self.encoder(**encoder_kwargs)
                )
            if isinstance(encoded, tuple):
                embeddings, attention_mask = encoded[0], encoded[1]
            else:
                embeddings = encoded
            embeddings = embeddings.detach()
        return self.value_model(embeddings, attention_mask)

    def expected_value(
        self,
        embeddings: Tensor | None = None,
        attention_mask: Tensor | None = None,
        *,
        encoder_inputs: tuple | None = None,
        **encoder_kwargs,
    ) -> Tensor:
        logits = self(
            embeddings,
            attention_mask,
            encoder_inputs=encoder_inputs,
            **encoder_kwargs,
        )
        return self.value_model.value_head.expected_value_from_logits(logits)


class StateTaskValueModel(nn.Module):
    """Small independent RECAP baseline conditioned on state and task ID.

    This model makes the first rollout/value/labeling loop runnable before a
    shared visual-language value encoder is available. It must be treated as a
    baseline: tasks whose progress is not observable from proprioception need a
    visual encoder in later RECAP iterations.
    """

    uses_task_id = True

    def __init__(
        self,
        state_dim: int,
        num_tasks: int,
        *,
        hidden_size: int = 256,
        task_embedding_dim: int = 64,
        num_bins: int = 201,
        value_min: float = -2000.0,
        value_max: float = 0.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive, got {num_tasks}")
        self.state_dim = int(state_dim)
        self.num_tasks = int(num_tasks)
        self.task_embedding = nn.Embedding(num_tasks, task_embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + task_embedding_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.value_head = CategoricalValueHead(
            hidden_size,
            num_bins=num_bins,
            value_min=value_min,
            value_max=value_max,
        )

    @property
    def value_support(self) -> Tensor:
        return self.value_head.value_support

    def forward(self, state: Tensor, task_id: Tensor) -> Tensor:
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, got {state.shape[-1]}")
        task_embedding = self.task_embedding(task_id.long())
        context = self.encoder(torch.cat((state, task_embedding), dim=-1))
        return self.value_head(context)

    def expected_value(self, state: Tensor, task_id: Tensor) -> Tensor:
        return self.value_head.expected_value_from_logits(self(state, task_id))


def discretize_returns(returns: Tensor, value_support: Tensor) -> Tensor:
    """Map scalar returns to the nearest categorical support index."""

    if value_support.ndim != 1 or value_support.numel() < 2:
        raise ValueError("value_support must be a one-dimensional tensor with at least two bins")
    if not torch.all(value_support[1:] > value_support[:-1]):
        raise ValueError("value_support must be strictly increasing")
    if not torch.isfinite(returns).all():
        raise ValueError("returns contains NaN or infinity")

    support = value_support.to(device=returns.device, dtype=returns.dtype)
    insertion = torch.searchsorted(support, returns.contiguous())
    upper = insertion.clamp(max=support.numel() - 1)
    lower = (upper - 1).clamp(min=0)
    lower_distance = (returns - support[lower]).abs()
    upper_distance = (support[upper] - returns).abs()
    return torch.where(upper_distance < lower_distance, upper, lower).long()


def categorical_value_loss(
    logits: Tensor,
    returns: Tensor,
    value_support: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Cross-entropy loss for discretized Monte-Carlo return targets."""

    if logits.shape[:-1] != returns.shape:
        raise ValueError(
            "Value logits and return targets must share leading dimensions, got "
            f"{tuple(logits.shape)} and {tuple(returns.shape)}"
        )
    targets = discretize_returns(returns, value_support)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(returns)
    if valid_mask is None:
        return losses.mean()
    if valid_mask.shape != returns.shape:
        raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} does not match returns {tuple(returns.shape)}")
    mask = valid_mask.to(device=losses.device, dtype=losses.dtype)
    return (losses * mask).sum() / mask.sum().clamp(min=1.0)


def monte_carlo_returns(
    rewards: Tensor,
    *,
    terminated: Tensor | None = None,
    valid_mask: Tensor | None = None,
    gamma: float = 1.0,
    bootstrap_value: Tensor | float | None = None,
) -> Tensor:
    """Compute masked Monte-Carlo returns along the last dimension.

    Each leading index is treated as an independent trajectory. ``terminated``
    cuts bootstrapping after true environment terminals. Invalid padded steps
    receive zero and do not propagate a return into earlier episodes.
    """

    if rewards.ndim == 0:
        raise ValueError("rewards must have a time dimension")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards contains NaN or infinity")

    if terminated is None:
        terminated = torch.zeros_like(rewards, dtype=torch.bool)
    if valid_mask is None:
        valid_mask = torch.ones_like(rewards, dtype=torch.bool)
    if terminated.shape != rewards.shape or valid_mask.shape != rewards.shape:
        raise ValueError("terminated and valid_mask must match rewards shape")

    if bootstrap_value is None:
        running = torch.zeros_like(rewards[..., 0])
    else:
        running = torch.as_tensor(
            bootstrap_value,
            device=rewards.device,
            dtype=rewards.dtype,
        )
        running = torch.broadcast_to(running, rewards.shape[:-1]).clone()

    output = torch.zeros_like(rewards)
    discount = float(gamma)
    for step in range(rewards.shape[-1] - 1, -1, -1):
        valid = valid_mask[..., step]
        continuation = (~terminated[..., step]).to(rewards.dtype)
        candidate = rewards[..., step] + discount * continuation * running
        running = torch.where(valid, candidate, torch.zeros_like(candidate))
        output[..., step] = torch.where(valid, running, torch.zeros_like(running))
    return output


N_STEP_HORIZON_UNITS = ("decisions", "actions")


def n_step_returns(
    rewards: Tensor,
    values: Tensor,
    *,
    n: int,
    gamma: float = 1.0,
    terminated: Tensor | None = None,
    valid_mask: Tensor | None = None,
    durations: Tensor | None = None,
    horizon_unit: str = "actions",
) -> Tensor:
    """n-step returns used by original RECAP advantage labeling.

    ``G_t = r_t + γ^{d_t} r_{t+1} + … + γ^{D} V(o_{t+n})``, stopping at the
    first terminal. Missing future steps do not bootstrap.

    ``horizon_unit="decisions"`` counts JSONL/decision chunks. ``"actions"``
    (default) accumulates ``durations`` so ``n=50`` matches the paper's
    50-step action horizon on chunked rollouts.
    """

    if int(n) < 1:
        raise ValueError(f"n must be a positive integer, got {n}")
    unit = str(horizon_unit)
    if unit not in N_STEP_HORIZON_UNITS:
        raise ValueError(
            f"horizon_unit must be one of {N_STEP_HORIZON_UNITS}, got {horizon_unit!r}"
        )
    if rewards.shape != values.shape:
        raise ValueError(
            f"rewards and values must have the same shape, got {tuple(rewards.shape)} and {tuple(values.shape)}"
        )
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not torch.isfinite(rewards).all() or not torch.isfinite(values).all():
        raise ValueError("rewards and values must be finite")

    if terminated is None:
        terminated = torch.zeros_like(rewards, dtype=torch.bool)
    if valid_mask is None:
        valid_mask = torch.ones_like(rewards, dtype=torch.bool)
    if durations is None:
        durations = torch.ones_like(rewards, dtype=rewards.dtype)
    else:
        durations = durations.to(device=rewards.device, dtype=rewards.dtype)
    if terminated.shape != rewards.shape or valid_mask.shape != rewards.shape:
        raise ValueError("terminated and valid_mask must match rewards shape")
    if durations.shape != rewards.shape:
        raise ValueError("durations must match rewards shape")
    if not torch.isfinite(durations).all() or bool((durations < 0).any()):
        raise ValueError("durations must be finite and non-negative")
    if unit == "actions" and bool((durations <= 0).any()):
        raise ValueError("action-horizon n-step requires positive durations")

    horizon = int(n)
    time_steps = rewards.shape[-1]
    flat_rewards = rewards.reshape(-1, time_steps)
    flat_values = values.reshape(-1, time_steps)
    flat_terminated = terminated.reshape(-1, time_steps)
    flat_valid = valid_mask.reshape(-1, time_steps)
    flat_durations = durations.reshape(-1, time_steps)
    output = torch.zeros_like(flat_rewards)
    discount_base = float(gamma)
    batch = flat_rewards.shape[0]

    for time_index in range(time_steps):
        remaining = torch.zeros(batch, dtype=flat_rewards.dtype, device=flat_rewards.device)
        discount = torch.ones_like(remaining)
        alive = flat_valid[:, time_index]
        if unit == "decisions":
            for offset in range(horizon):
                index = time_index + offset
                if index >= time_steps:
                    break
                step_valid = alive & flat_valid[:, index]
                remaining = torch.where(
                    step_valid,
                    remaining + discount * flat_rewards[:, index],
                    remaining,
                )
                alive = step_valid & ~flat_terminated[:, index]
                discount = torch.where(
                    alive,
                    discount * torch.pow(discount_base, flat_durations[:, index]),
                    discount,
                )
            bootstrap_index = time_index + horizon
            if bootstrap_index < time_steps:
                bootstrap = alive & flat_valid[:, bootstrap_index]
                remaining = torch.where(
                    bootstrap,
                    remaining + discount * flat_values[:, bootstrap_index],
                    remaining,
                )
        else:
            elapsed = torch.zeros_like(remaining)
            for offset in range(time_steps - time_index):
                index = time_index + offset
                step_valid = alive & flat_valid[:, index]
                remaining = torch.where(
                    step_valid,
                    remaining + discount * flat_rewards[:, index],
                    remaining,
                )
                elapsed = torch.where(step_valid, elapsed + flat_durations[:, index], elapsed)
                alive = step_valid & ~flat_terminated[:, index]
                discount = torch.where(
                    alive,
                    discount * torch.pow(discount_base, flat_durations[:, index]),
                    discount,
                )
                reached = alive & (elapsed >= float(horizon))
                next_index = index + 1
                if next_index < time_steps:
                    bootstrap = reached & flat_valid[:, next_index]
                    remaining = torch.where(
                        bootstrap,
                        remaining + discount * flat_values[:, next_index],
                        remaining,
                    )
                alive = alive & (elapsed < float(horizon))
                if not bool(alive.any()):
                    break
        output[:, time_index] = torch.where(flat_valid[:, time_index], remaining, torch.zeros_like(remaining))
    return output.reshape(rewards.shape)


class VisualTaskValueModel(nn.Module):
    """Distributional critic over stacked camera images and a task id.

    This is the visual replacement for :class:`StateTaskValueModel`. It does
    not load the 6B VLA; callers that already have a pooled VLM embedding
    should use :class:`VLMPooledValueModel`. Images are ``[B, V, C, H, W]`` in
    ``[0, 1]``.
    """

    uses_task_id = True

    def __init__(
        self,
        num_tasks: int,
        *,
        in_channels: int = 3,
        image_size: int = 64,
        hidden_size: int = 256,
        task_embedding_dim: int = 64,
        encoder_width: int = 64,
        num_bins: int = 201,
        value_min: float = -2000.0,
        value_max: float = 0.0,
    ) -> None:
        super().__init__()
        if num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive, got {num_tasks}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.num_tasks = int(num_tasks)
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.task_embedding = nn.Embedding(num_tasks, task_embedding_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, encoder_width // 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(encoder_width // 2, encoder_width, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.visual_projection = nn.Linear(encoder_width, hidden_size)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size + task_embedding_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.value_head = CategoricalValueHead(
            hidden_size,
            num_bins=num_bins,
            value_min=value_min,
            value_max=value_max,
        )

    @property
    def value_support(self) -> Tensor:
        return self.value_head.value_support

    def _encode_images(self, images: Tensor) -> Tensor:
        if images.ndim != 5:
            raise ValueError(f"images must have shape [B, V, C, H, W], got {tuple(images.shape)}")
        batch, views, channels, height, width = images.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} image channels, got {channels}")
        if views <= 0:
            raise ValueError("images must contain at least one camera view")
        flat = images.reshape(batch * views, channels, height, width).to(dtype=torch.float32)
        if height != self.image_size or width != self.image_size:
            flat = F.interpolate(
                flat,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        features = self.encoder(flat).flatten(1)
        projected = self.visual_projection(features).reshape(batch, views, -1)
        return projected.mean(dim=1)

    def forward(self, images: Tensor, task_id: Tensor) -> Tensor:
        visual = self._encode_images(images)
        task_embedding = self.task_embedding(task_id.long())
        context = self.trunk(torch.cat((visual, task_embedding), dim=-1))
        return self.value_head(context)

    def expected_value(self, images: Tensor, task_id: Tensor) -> Tensor:
        return self.value_head.expected_value_from_logits(self(images, task_id))
