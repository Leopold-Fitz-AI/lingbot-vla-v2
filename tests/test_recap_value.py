import unittest

import torch

from lingbotvla.recap.labels import (
    RecapAdvantageLabel,
    compute_advantages,
    label_advantages,
    n_step_advantages,
    positive_quantile_threshold,
    time_to_success_rewards,
)
from lingbotvla.recap.value import (
    CategoricalValueHead,
    VisualTaskValueModel,
    categorical_value_loss,
    discretize_returns,
    monte_carlo_returns,
    n_step_returns,
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


class NStepReturnTest(unittest.TestCase):
    def test_one_step_bootstraps_the_next_value(self):
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        values = torch.tensor([[10.0, 20.0, 30.0]])
        returns = n_step_returns(rewards, values, n=1, gamma=1.0)
        self.assertTrue(torch.allclose(returns, torch.tensor([[21.0, 32.0, 3.0]])))

    def test_two_step_return_matches_paper_formula(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([10.0, 20.0, 30.0])
        returns = n_step_returns(rewards, values, n=2, gamma=1.0)
        self.assertTrue(torch.allclose(returns, torch.tensor([1.0 + 2.0 + 30.0, 2.0 + 3.0, 3.0])))

    def test_gamma_discounts_bootstrap_and_intermediate_reward(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.0, 4.0, 0.0])
        returns = n_step_returns(rewards, values, n=2, gamma=0.5)
        self.assertTrue(torch.allclose(returns, torch.tensor([1.0 + 0.5 * 2.0 + 0.25 * 0.0, 2.0 + 0.5 * 3.0, 3.0])))

    def test_termination_cuts_bootstrap(self):
        rewards = torch.tensor([1.0, 2.0, 99.0])
        values = torch.tensor([0.0, 50.0, 0.0])
        terminated = torch.tensor([False, True, False])
        returns = n_step_returns(rewards, values, n=2, terminated=terminated)
        self.assertTrue(torch.allclose(returns, torch.tensor([1.0 + 2.0, 2.0, 99.0])))

    def test_durations_scale_the_discount(self):
        rewards = torch.tensor([1.0, 2.0])
        values = torch.tensor([0.0, 10.0])
        durations = torch.tensor([2, 1])
        returns = n_step_returns(rewards, values, n=1, gamma=0.5, durations=durations)
        self.assertTrue(torch.allclose(returns, torch.tensor([1.0 + 0.25 * 10.0, 2.0])))

    def test_n_step_advantage_subtracts_current_value(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([10.0, 20.0, 30.0])
        advantages = n_step_advantages(rewards, values, n=1)
        self.assertTrue(torch.allclose(advantages, torch.tensor([11.0, 12.0, -27.0])))

    def test_action_horizon_spans_chunks_until_n_actions(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([10.0, 20.0, 30.0])
        durations = torch.tensor([30.0, 30.0, 10.0])
        returns = n_step_returns(
            rewards,
            values,
            n=50,
            durations=durations,
            horizon_unit="actions",
        )
        self.assertTrue(torch.allclose(returns, torch.tensor([1.0 + 2.0 + 30.0, 2.0 + 3.0, 3.0])))

    def test_action_horizon_stops_after_one_full_chunk(self):
        rewards = torch.tensor([1.0, 2.0])
        values = torch.tensor([0.0, 10.0])
        durations = torch.tensor([50.0, 10.0])
        returns = n_step_returns(
            rewards,
            values,
            n=50,
            gamma=0.5,
            durations=durations,
            horizon_unit="actions",
        )
        self.assertTrue(torch.allclose(returns, torch.tensor([1.0 + (0.5**50) * 10.0, 2.0])))

    def test_decision_horizon_ignores_action_durations_for_span(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([10.0, 20.0, 30.0])
        durations = torch.tensor([30.0, 30.0, 10.0])
        returns = n_step_returns(
            rewards,
            values,
            n=1,
            durations=durations,
            horizon_unit="decisions",
        )
        self.assertTrue(torch.allclose(returns, torch.tensor([1.0 + 20.0, 2.0 + 30.0, 3.0])))


class QuantileThresholdTest(unittest.TestCase):
    def test_positive_quantile_keeps_the_requested_mass(self):
        advantages = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        threshold = positive_quantile_threshold(advantages, positive_fraction=0.5)
        labels = label_advantages(advantages, threshold=threshold)
        self.assertEqual(int((labels == RecapAdvantageLabel.POSITIVE).sum()), 3)


class VisualTaskValueTest(unittest.TestCase):
    def test_visual_model_maps_cameras_and_task_to_value_bins(self):
        model = VisualTaskValueModel(
            num_tasks=2,
            in_channels=3,
            image_size=8,
            hidden_size=16,
            task_embedding_dim=4,
            num_bins=5,
            value_min=-4,
            value_max=0,
        )
        images = torch.zeros(2, 2, 3, 8, 8)
        images[1] = 1.0
        logits = model(images, torch.tensor([0, 1]))
        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertEqual(tuple(model.expected_value(images, torch.tensor([0, 1])).shape), (2,))
        loss = categorical_value_loss(
            logits,
            torch.tensor([-2.0, 0.0]),
            model.value_support,
        )
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


class InterventionAndNStepLabelTest(unittest.TestCase):
    def test_n_step_labels_use_bootstrapped_advantage(self):
        episode = {
            "episode_id": "demo/n-step",
            "task": "pick cup",
            "success": True,
            "steps": [
                {"value": 0.0, "reward": -1.0, "terminated": False},
                {"value": -1.0, "reward": 0.0, "terminated": True},
            ],
        }
        labeled = label_episode(episode, n_step=1)
        self.assertAlmostEqual(labeled["steps"][0]["recap_advantage"], -2.0)
        self.assertEqual(labeled["steps"][0]["recap_label"], int(RecapAdvantageLabel.NEGATIVE))
        self.assertAlmostEqual(labeled["steps"][1]["recap_advantage"], 1.0)
        self.assertEqual(labeled["steps"][1]["recap_label"], int(RecapAdvantageLabel.POSITIVE))

    def test_action_horizon_labels_span_executed_action_lengths(self):
        episode = {
            "episode_id": "demo/chunks",
            "task": "pick cup",
            "success": True,
            "steps": [
                {
                    "value": 0.0,
                    "reward": -30.0,
                    "terminated": False,
                    "executed_action_length": 30,
                },
                {
                    "value": -10.0,
                    "reward": -20.0,
                    "terminated": False,
                    "executed_action_length": 30,
                },
                {
                    "value": 0.0,
                    "reward": 0.0,
                    "terminated": True,
                    "executed_action_length": 10,
                },
            ],
        }
        labeled = label_episode(episode, n_step=50, n_step_unit="actions")
        self.assertAlmostEqual(labeled["steps"][0]["recap_advantage"], -50.0)
        self.assertEqual(labeled["steps"][0]["recap_label"], int(RecapAdvantageLabel.NEGATIVE))
        self.assertAlmostEqual(labeled["steps"][1]["recap_advantage"], -10.0)
        self.assertEqual(labeled["steps"][1]["recap_label"], int(RecapAdvantageLabel.NEGATIVE))

    def test_intervention_steps_are_forced_positive(self):
        episode = {
            "episode_id": "demo/intervene",
            "task": "pick cup",
            "success": False,
            "steps": [
                {"value": 0.0, "reward": -1.0, "terminated": False, "intervention": True},
                {"value": 0.0, "reward": -1000.0, "terminated": True},
            ],
        }
        labeled = label_episode(episode, failure_penalty=1000)
        self.assertEqual(labeled["steps"][0]["recap_label"], int(RecapAdvantageLabel.POSITIVE))
        self.assertEqual(labeled["steps"][1]["recap_label"], int(RecapAdvantageLabel.NEGATIVE))


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
