import random

import numpy as np
import pytest
import torch

from lingbotvla.recap.adapter import apply_recap_velocity_lora
from lingbotvla.recap.initialization import initialize_velocity_lora_


def test_full_rank_has_matched_norm_and_preserves_all_rngs():
    a, b = torch.empty(2, 8, 768), torch.empty(2, 55, 8)
    rng = torch.get_rng_state().clone()
    py_rng, np_rng = random.getstate(), np.random.get_state()
    metadata = initialize_velocity_lora_(a, b, scheme="orthogonal_matched_v1", seed=971)
    assert torch.equal(rng, torch.get_rng_state())
    assert py_rng == random.getstate()
    assert all(np.array_equal(x, y) for x, y in zip(np_rng, np.random.get_state()))
    assert torch.count_nonzero(b) == 0
    original = (torch.sin(torch.arange(a.numel()).float().reshape(a.shape) * 0.017) * 0.02).double()
    for i in range(2):
        sv = torch.linalg.svdvals(a[i].double())
        assert sv[-1] / sv[0] > 0.99999
        assert torch.linalg.vector_norm(a[i].double()).item() == pytest.approx(
            torch.linalg.vector_norm(original[i]).item(), rel=1e-7
        )
    repeat = torch.empty_like(a)
    initialize_velocity_lora_(repeat, b, scheme="orthogonal_matched_v1", seed=971)
    assert torch.equal(a, repeat)
    initialize_velocity_lora_(repeat, b, scheme="orthogonal_matched_v1", seed=972)
    assert not torch.equal(a, repeat)
    assert metadata["generator_device"] == "cpu"
    assert metadata["target_dtype"] == "torch.float32"


def test_new_initialization_zero_behavior_and_gradient_flow():
    a = torch.nn.Parameter(torch.empty(2, 8, 24))
    b = torch.nn.Parameter(torch.empty(2, 14, 8))
    initialize_velocity_lora_(a, b, scheme="orthogonal_matched_v1", seed=971)
    hidden = torch.randn(3, 5, 24)
    optimizer = torch.optim.SGD([a, b], lr=0.5)
    residual = apply_recap_velocity_lora(hidden, [-1, 0, 1], a, b, signed_axis=True)
    assert torch.count_nonzero(residual) == 0
    (residual - 1).square().mean().backward()
    assert b.grad.abs().max() > 0 and torch.count_nonzero(a.grad) == 0
    optimizer.step()
    optimizer.zero_grad()
    residual = apply_recap_velocity_lora(hidden, [-1, 0, 1], a, b, signed_axis=True)
    assert torch.count_nonzero(residual[0]) == 0
    (residual - 1).square().mean().backward()
    assert a.grad.abs().max() > 0


def test_legacy_arithmetic_is_unchanged():
    a, b = torch.empty(2, 8, 768), torch.empty(2, 55, 8)
    initialize_velocity_lora_(a, b)
    expected = torch.sin(torch.arange(a.numel(), dtype=torch.float32).reshape_as(a) * 0.017) * 0.02
    assert torch.equal(a, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scheme": "unknown"},
        {"seed": True},
        {"seed": -1},
        {"init_std": float("nan")},
        {"init_std": float("inf")},
        {"scheme": "orthogonal_matched_v1", "init_std": 0},
    ],
)
def test_invalid_initialization_rejected(kwargs):
    with pytest.raises(ValueError):
        initialize_velocity_lora_(torch.empty(2, 2, 4), torch.empty(2, 3, 2), **kwargs)


def test_full_row_rank_cannot_exceed_hidden_width():
    with pytest.raises(ValueError, match="rank <= hidden"):
        initialize_velocity_lora_(torch.empty(2, 8, 4), torch.empty(2, 3, 8), scheme="orthogonal_matched_v1")
