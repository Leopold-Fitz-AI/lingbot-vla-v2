import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lingbotvla.recap.ablation import (
    assert_training_sources,
    cached_losses,
    checked_gradients,
    draw_seed,
    sample_schedule,
    summarize_pairs,
)
from lingbotvla.recap.adapter import apply_recap_velocity_lora
from lingbotvla.recap.initialization import initialize_velocity_lora_
from lingbotvla.recap.loss import masked_action_loss
from scripts.recap_run_training_ablation import null_gate, save_adapter, save_bytes


def test_schedule_is_complete_drop_last_and_does_not_use_global_rng():
    np.random.seed(47)
    before = np.random.get_state()
    a = sample_schedule(26, 2101)
    assert len(a) == 50 and all(len(row) == 4 for row in a)
    assert a == sample_schedule(26, 2101) and a != sample_schedule(26, 2102)
    assert all(len(set(sum(a[i : i + 6], []))) == 24 for i in range(0, 48, 6))
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1])
    assert len({draw_seed("mug", 2101, i, j) for i in range(50) for j in range(4)}) == 200


@pytest.mark.parametrize("path", ["/data/final/foo", "/data/cohorts/s51/foo", "/data/s50/x", "/data/s52/x"])
def test_no_final_training_sources(path):
    with pytest.raises(ValueError, match="Final data"):
        assert_training_sources([path])


@pytest.mark.parametrize("weight", [0.0, 100.0])
def test_cached_loss_and_gradient_match_actual_policy_wrapper(weight):
    path = Path(__file__).resolve().parents[1] / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    cls = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "LingbotVlaV2Policy"
    )
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    scope = {"torch": torch, "Tensor": torch.Tensor, "masked_action_loss": masked_action_loss}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(path), "exec"), scope)
    a, b = nn.Parameter(torch.randn(2, 2, 4)), nn.Parameter(torch.randn(2, 3, 2))
    hidden, base = torch.randn(1, 5, 4), torch.randn(1, 5, 3)
    actions, noise = torch.randn(1, 5, 3), torch.randn(1, 5, 3)
    mask = torch.ones_like(actions, dtype=torch.bool)
    mask[:, :, 2] = False
    pad = torch.tensor([[False, False, False, True, True]])

    class Inner(nn.Module):
        def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise, time, **kwargs):
            self._last_recap_velocity_residual = apply_recap_velocity_lora(
                hidden, kwargs["recap_condition_id"], a, b, scale=8.0, signed_axis=True
            )
            loss = torch.nn.functional.l1_loss(
                noise - actions, base + self._last_recap_velocity_residual, reduction="none"
            )
            return loss, 0.0, 0.0, 0.0, None, 0.0, 0.0, {}, None, None, None

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.config = SimpleNamespace(action_dim=3, loss_type="L1_fm", recap_residual_loss_weight=weight)

    Policy.forward = scope["forward"]
    actual = Policy()(
        None, None, None, None, None, actions, joint_mask=mask, action_is_pad=pad, noise=noise, recap_condition_id=[0]
    )[0]
    cached = cached_losses(hidden, base, actions, noise, [0], a, b, mask, pad, scale=8.0, weight=weight)[0]
    assert torch.equal(actual, cached)
    assert all(
        torch.equal(l, r) for l, r in zip(torch.autograd.grad(actual, [a, b]), torch.autograd.grad(cached, [a, b]))
    )


def test_gradient_initialization_null_and_compact_export(tmp_path):
    a, b = nn.Parameter(torch.empty(2, 2, 4)), nn.Parameter(torch.empty(2, 3, 2))
    initialize_velocity_lora_(a, b, scheme="orthogonal_matched_v1")
    h, v = torch.randn(1, 5, 4), torch.randn(1, 5, 3)
    null_gate(a, b, h, v, 8.0, initial=True)
    opt = torch.optim.AdamW([a, b], lr=0.001, weight_decay=0.0)
    for step in range(2):
        opt.zero_grad()
        loss = cached_losses(
            h, v, torch.zeros_like(v), torch.ones_like(v), [1], a, b, None, None, scale=8.0, weight=0.0
        )[0]
        loss.backward()
        checked_gradients([a, b], step=step)
        opt.step()
        null_gate(a, b, h, v, 8.0)
    artifact = save_adapter(tmp_path, [a, b], {"test": True})
    assert Path(artifact["path"]).is_file()
    with pytest.raises(FileExistsError):
        save_adapter(tmp_path, [a, b], {})
    a.grad.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        checked_gradients([a, b], step=2)


def test_no_overwriting_outputs(tmp_path):
    path = tmp_path / "data"
    save_bytes(path, b"old")
    with pytest.raises(FileExistsError):
        save_bytes(path, b"new")
    assert path.read_bytes() == b"old"


def test_surrogates_count_states_not_noise_draws_and_require_pairs():
    rows = []
    for noise in range(12):
        for label in (0, 1):
            rows.append(
                dict(
                    split="holdout",
                    seed=3,
                    pair_id="p1",
                    time=0.5,
                    noise_seed=noise,
                    label=label,
                    base_fm=1.0,
                    positive_fm=1.0 - 0.1 * label,
                )
            )
    summary = summarize_pairs(rows)["holdout"]
    assert summary["states"] == 1 and summary["paired_draws"] == 12
    assert summary["surrogate_margin_mean"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_pairs(rows[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_pairs(rows + [rows[0]])
    json.dumps(summary, allow_nan=False)
