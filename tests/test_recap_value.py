import unittest

import torch

from lingbotvla.recap.labels import (
    RecapAdvantageLabel,
    compute_advantages,
    label_advantages,
    time_to_success_rewards,
)
from lingbotvla.recap.value import (
    CategoricalValueHead,
    categorical_value_loss,
    discretize_returns,
    monte_carlo_returns,
)
from scripts.recap_label_rollouts import label_episode
from tasks.vla.train_recap_value import _split_episodes


class MonteCarloReturnTest(unittest.TestCase):
    def test_time_to_success_success_and_failure(self):
        success_rewards = time_to_success_rewards(3, success=True, failure_penalty=10)
        failure_rewards = time_to_success_rewards(3, success=False, failure_penalty=10)
        self.assertTrue(torch.equal(success_rewards, torch.tensor([-1.0, -1.0, 0.0])))
        self.assertTrue(torch.equal(failure_rewards, torch.tensor([-1.0, -1.0, -10.0])))
        self.assertTrue(
            torch.equal(
                monte_carlo_returns(success_rewards),
                torch.tensor([-2.0, -1.0, 0.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                monte_carlo_returns(failure_rewards),
                torch.tensor([-12.0, -11.0, -10.0]),
            )
        )

    def test_terminated_cuts_future_return(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        terminated = torch.tensor([False, True, False])
        returns = monte_carlo_returns(rewards, terminated=terminated)
        self.assertTrue(torch.equal(returns, torch.tensor([3.0, 2.0, 3.0])))

    def test_padding_does_not_leak_into_return(self):
        rewards = torch.tensor([1.0, 2.0, 99.0])
        valid = torch.tensor([True, True, False])
        returns = monte_carlo_returns(rewards, valid_mask=valid)
        self.assertTrue(torch.equal(returns, torch.tensor([3.0, 2.0, 0.0])))


class CategoricalValueTest(unittest.TestCase):
    def test_head_shape_and_expected_value(self):
        head = CategoricalValueHead(4, num_bins=3, value_min=-2, value_max=0)
        inputs = torch.randn(2, 4)
        logits = head(inputs)
        self.assertEqual(tuple(logits.shape), (2, 3))
        centered = head.expected_value_from_logits(torch.zeros(2, 3))
        self.assertTrue(torch.allclose(centered, torch.tensor([-1.0, -1.0])))

    def test_discretizes_to_nearest_bin_and_clamps(self):
        support = torch.tensor([-2.0, -1.0, 0.0])
        returns = torch.tensor([-5.0, -1.2, -0.1, 3.0])
        indices = discretize_returns(returns, support)
        self.assertTrue(torch.equal(indices, torch.tensor([0, 1, 2, 2])))

    def test_masked_value_loss_ignores_invalid_targets(self):
        support = torch.tensor([-1.0, 0.0, 1.0])
        logits = torch.tensor(
            [[8.0, 0.0, 0.0], [0.0, 0.0, 8.0]],
            requires_grad=True,
        )
        returns = torch.tensor([-1.0, -1.0])
        masked = categorical_value_loss(
            logits,
            returns,
            support,
            valid_mask=torch.tensor([True, False]),
        )
        unmasked = categorical_value_loss(logits, returns, support)
        self.assertLess(masked.item(), unmasked.item())
        masked.backward()
        self.assertTrue(torch.equal(logits.grad[1], torch.zeros(3)))


class EpisodeSplitTest(unittest.TestCase):
    def test_stratified_split_keeps_success_and_failure_in_both_sets(self):
        episode_ids = [f"success-{index}" for index in range(10)] + [
            f"failure-{index}" for index in range(4)
        ]
        strata = {episode_id: episode_id.startswith("success") for episode_id in episode_ids}
        train, validation = _split_episodes(
            episode_ids,
            validation_fraction=0.2,
            seed=0,
            episode_strata=strata,
        )
        self.assertEqual(len(validation), 3)
        self.assertEqual(len(train), 11)
        self.assertEqual(sum(strata[item] for item in validation), 2)
        self.assertEqual(sum(not strata[item] for item in validation), 1)
        self.assertTrue(any(strata[item] for item in train))
        self.assertTrue(any(not strata[item] for item in train))
        self.assertFalse(train & validation)

    def test_stratified_split_is_deterministic(self):
        episodes = ["success-a", "success-b", "failure-a", "failure-b"]
        strata = {item: item.startswith("success") for item in episodes}
        first = _split_episodes(episodes, validation_fraction=0.25, seed=7, episode_strata=strata)
        second = _split_episodes(episodes, validation_fraction=0.25, seed=7, episode_strata=strata)
        self.assertEqual(first, second)


class RolloutLabelingTest(unittest.TestCase):
    def test_labels_episode_from_value_predictions(self):
        episode = {
            "episode_id": "demo/0",
            "task": "pick cup",
            "success": True,
            "steps": [
                {"value": -3.0},
                {"value": 0.0},
                {"value": 1.0},
            ],
        }
        labeled = label_episode(episode, failure_penalty=10)
        self.assertEqual(
            [step["recap_return"] for step in labeled["steps"]],
            [-2.0, -1.0, 0.0],
        )
        self.assertEqual(
            [step["recap_label"] for step in labeled["steps"]],
            [
                int(RecapAdvantageLabel.POSITIVE),
                int(RecapAdvantageLabel.NEGATIVE),
                int(RecapAdvantageLabel.NEGATIVE),
            ],
        )

    def test_outcome_mode_never_marks_failed_episode_positive(self):
        failed = {
            "episode_id": "demo/failed",
            "task": "click bell",
            "success": False,
            "steps": [
                {"value": -1001.0},
                {"value": -1002.0},
                {"value": -1003.0, "valid": False},
            ],
        }
        labeled = label_episode(failed, failure_penalty=1000, label_mode="outcome")
        self.assertEqual(
            [step["recap_label"] for step in labeled["steps"]],
            [
                int(RecapAdvantageLabel.NEGATIVE),
                int(RecapAdvantageLabel.NEGATIVE),
                int(RecapAdvantageLabel.NEUTRAL),
            ],
        )
        self.assertIn("recap_advantage", labeled["steps"][0])

    def test_outcome_mode_marks_successful_episode_positive(self):
        succeeded = {
            "episode_id": "demo/succeeded",
            "task": "click bell",
            "success": True,
            "steps": [{}, {}],
        }
        labeled = label_episode(succeeded, failure_penalty=1000, label_mode="outcome")
        self.assertEqual(
            [step["recap_label"] for step in labeled["steps"]],
            [int(RecapAdvantageLabel.POSITIVE)] * 2,
        )


class AdvantageLabelTest(unittest.TestCase):
    def test_labels_positive_negative_and_neutral(self):
        advantages = compute_advantages(
            torch.tensor([2.0, 1.0, 0.0]),
            torch.tensor([1.0, 1.0, 1.0]),
        )
        labels = label_advantages(advantages, neutral_margin=0.25)
        expected = torch.tensor(
            [
                int(RecapAdvantageLabel.POSITIVE),
                int(RecapAdvantageLabel.NEUTRAL),
                int(RecapAdvantageLabel.NEGATIVE),
            ],
            dtype=torch.int8,
        )
        self.assertTrue(torch.equal(labels, expected))

    def test_invalid_values_are_neutral(self):
        labels = label_advantages(
            torch.tensor([1.0, -1.0]),
            valid_mask=torch.tensor([False, True]),
        )
        self.assertTrue(
            torch.equal(
                labels,
                torch.tensor(
                    [
                        int(RecapAdvantageLabel.NEUTRAL),
                        int(RecapAdvantageLabel.NEGATIVE),
                    ],
                    dtype=torch.int8,
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
