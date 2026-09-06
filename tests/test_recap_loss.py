import torch

from lingbotvla.recap.loss import masked_action_loss


def test_action_padding_is_excluded_from_flow_loss():
    losses = torch.tensor([[[1.0, 3.0], [100.0, 200.0]]])
    loss, per_sample = masked_action_loss(
        losses,
        action_dim=2,
        action_is_pad=torch.tensor([[False, True]]),
    )
    assert torch.equal(loss, torch.tensor(2.0))
    assert torch.equal(per_sample, torch.tensor([2.0]))


def test_padding_combines_with_joint_mask():
    losses = torch.tensor([[[1.0, 99.0], [100.0, 200.0]]])
    joint_mask = torch.tensor([[[True, False], [True, False]]])
    loss, per_sample = masked_action_loss(
        losses,
        action_dim=2,
        joint_mask=joint_mask,
        action_is_pad=torch.tensor([[False, True]]),
    )
    assert torch.equal(loss, torch.tensor(1.0))
    assert torch.equal(per_sample, torch.tensor([1.0]))


def test_repeated_loss_repeats_both_masks():
    losses = torch.tensor(
        [
            [[1.0], [100.0]],
            [[3.0], [200.0]],
        ]
    )
    loss, per_sample = masked_action_loss(
        losses,
        action_dim=1,
        joint_mask=torch.ones((1, 2, 1), dtype=torch.bool),
        action_is_pad=torch.tensor([[False, True]]),
        repeated_loss=True,
    )
    assert torch.equal(loss, torch.tensor(2.0))
    assert torch.equal(per_sample, torch.tensor([1.0, 3.0]))
