"""Prompt conditioning primitives for RECAP.

RECAP trains one policy on unconditional, positive-advantage, and
negative-advantage examples.  Keeping prompt construction here makes training
and deployment use exactly the same textual convention.
"""

from __future__ import annotations

from enum import Enum
from numbers import Integral, Real
from typing import Any, Callable


class RecapCondition(str, Enum):
    """Supported RECAP policy conditions."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NULL = "null"


_POSITIVE_ALIASES = {
    "1",
    "good",
    "high",
    "improved",
    "positive",
    "pos",
    "success",
    "true",
}
_NEGATIVE_ALIASES = {
    "0",
    "bad",
    "failure",
    "false",
    "low",
    "negative",
    "neg",
}
_NULL_ALIASES = {"", "none", "null", "unconditional", "unknown"}


def _to_python_scalar(value: Any) -> Any:
    """Convert scalar tensor/array-like values without importing their packages."""

    if value is None or isinstance(value, (str, bool, Integral, Real, RecapCondition)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError, RuntimeError):
            pass
    return value


def normalize_recap_condition(
    value: Any,
    *,
    missing: str | RecapCondition = RecapCondition.NULL,
) -> RecapCondition:
    """Normalize a dataset value into a :class:`RecapCondition`.

    Args:
        value: Boolean, -1/0/1 scalar, string label, or ``None``.
        missing: Condition used when ``value`` is ``None``. Use ``"error"`` to
            require every sample to carry an explicit label.
    """

    value = _to_python_scalar(value)
    if value is None:
        if str(missing).lower() == "error":
            raise ValueError("Missing RECAP advantage condition")
        value = missing

    if isinstance(value, RecapCondition):
        return value
    if isinstance(value, bool):
        return RecapCondition.POSITIVE if value else RecapCondition.NEGATIVE
    if isinstance(value, Integral):
        if int(value) == 1:
            return RecapCondition.POSITIVE
        if int(value) == 0:
            return RecapCondition.NEGATIVE
        if int(value) == -1:
            return RecapCondition.NULL
        raise ValueError(f"RECAP integer condition must be -1, 0, or 1, got {value!r}")
    if isinstance(value, Real):
        numeric = float(value)
        if numeric == 1.0:
            return RecapCondition.POSITIVE
        if numeric == 0.0:
            return RecapCondition.NEGATIVE
        if numeric == -1.0:
            return RecapCondition.NULL
        raise ValueError(f"RECAP numeric condition must be -1, 0, or 1, got {value!r}")

    normalized = str(value).strip().lower()
    if normalized.startswith("recapcondition."):
        normalized = normalized.split(".", 1)[1]
    if normalized in _POSITIVE_ALIASES:
        return RecapCondition.POSITIVE
    if normalized in _NEGATIVE_ALIASES:
        return RecapCondition.NEGATIVE
    if normalized in _NULL_ALIASES:
        return RecapCondition.NULL
    raise ValueError(
        f"Unsupported RECAP condition {value!r}; expected positive, negative, null, true/false, or -1/0/1"
    )


def recap_condition_id(
    condition: RecapCondition | str | int | float | bool | None,
    *,
    missing: str | RecapCondition = RecapCondition.NULL,
) -> int:
    """Return the model-facing condition id: null=-1, negative=0, positive=1."""

    normalized = normalize_recap_condition(condition, missing=missing)
    if normalized is RecapCondition.NULL:
        return -1
    if normalized is RecapCondition.NEGATIVE:
        return 0
    return 1


def maybe_drop_recap_condition(
    condition: RecapCondition,
    dropout_probability: float,
    *,
    random_value: float | None = None,
    random_fn: Callable[[], float] | None = None,
) -> RecapCondition:
    """Drop a condition to train the unconditional policy branch.

    ``random_value`` exists so unit tests and deterministic callers do not need
    to manipulate global random state.  Data loading normally passes
    ``random_fn=torch.rand`` indirectly from ``FeatureTransform``.
    """

    probability = float(dropout_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"RECAP condition dropout must be in [0, 1], got {probability}")
    if condition is RecapCondition.NULL or probability == 0.0:
        return condition
    if random_value is None:
        if random_fn is None:
            import random

            random_value = random.random()
        else:
            random_value = float(random_fn())
    return RecapCondition.NULL if float(random_value) < probability else condition


def format_recap_prompt(
    task: str,
    condition: RecapCondition | str,
    *,
    condition_name: str = "Advantage",
) -> str:
    """Build the shared training/inference prompt.

    The condition is prepended so tokenizer truncation cannot silently remove
    the policy-control signal from long task descriptions.  The null branch is
    truly unconditional: no synthetic ``null`` text is inserted.
    """

    task = str(task).strip()
    normalized = normalize_recap_condition(condition)
    if normalized is RecapCondition.NULL:
        return task
    return f"{condition_name}: {normalized.value}.\nTask: {task}"
