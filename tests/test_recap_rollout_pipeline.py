import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from deploy.recap_rollout_recorder import RecapEpisodeRecorder
from lingbotvla.recap.rollouts import (
    decisions_to_json_episode,
    load_decision_state,
    load_rollout_decisions,
)
from lingbotvla.recap.value import StateTaskValueModel


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def record_episode(root, episode_id, *, success, state_offset=0):
    recorder = RecapEpisodeRecorder(
        root,
        episode_id=episode_id,
        task="pick cup",
        seed=state_offset,
        metadata={"task_name": "pick_cup"},
    )
    for decision_index, action_length in enumerate((3, 2)):
        action = np.full((action_length, 2), decision_index, dtype=np.float32)
        recorder.record_step(
            {"observation.state": np.array([state_offset, decision_index], dtype=np.float32)},
            action,
            action,
            terminated=success and decision_index == 1,
            truncated=(not success and decision_index == 1),
        )
    recorder.finalize(
        success=success,
        terminal_reason="success" if success else "step_limit",
    )


class RolloutLoadingTest(unittest.TestCase):
    def test_duration_aware_rewards_and_state_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            record_episode(directory, "success", success=True)
            decisions = load_rollout_decisions(directory, failure_penalty=10)
            self.assertEqual([item.reward for item in decisions], [-3.0, -1.0])
            self.assertEqual([item.empirical_return for item in decisions], [-4.0, -1.0])
            self.assertTrue(
                np.array_equal(
                    load_decision_state(decisions[1]),
                    np.array([0.0, 1.0], dtype=np.float32),
                )
            )
            episode = decisions_to_json_episode(decisions, [-3.5, -0.5])
            self.assertTrue(episode["success"])
            self.assertEqual(episode["steps"][0]["executed_action_length"], 3)

    def test_failure_penalty_replaces_terminal_time_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            record_episode(directory, "failure", success=False)
            decisions = load_rollout_decisions(directory, failure_penalty=10)
            self.assertEqual([item.reward for item in decisions], [-3.0, -11.0])
            self.assertEqual([item.empirical_return for item in decisions], [-14.0, -11.0])

    def test_discounting_uses_executed_action_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            record_episode(directory, "success", success=True)
            decisions = load_rollout_decisions(directory, failure_penalty=10, gamma=0.5)
            # First chunk has three -1 rewards: -1 - .5 - .25. The final
            # two-action success chunk contains one -1 and then terminal 0.
            self.assertAlmostEqual(decisions[0].reward, -1.75)
            self.assertAlmostEqual(decisions[1].reward, -1.0)
            self.assertAlmostEqual(decisions[0].empirical_return, -1.875)


class StateTaskValueModelTest(unittest.TestCase):
    def test_model_outputs_distribution_and_value(self):
        model = StateTaskValueModel(
            state_dim=4,
            num_tasks=2,
            hidden_size=8,
            task_embedding_dim=3,
            num_bins=5,
            value_min=-4,
            value_max=0,
        )
        state = torch.randn(3, 4)
        task_id = torch.tensor([0, 1, 0])
        self.assertEqual(tuple(model(state, task_id).shape), (3, 5))
        self.assertEqual(tuple(model.expected_value(state, task_id).shape), (3,))


class ValuePipelineIntegrationTest(unittest.TestCase):
    def test_train_predict_and_label_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout_dir = root / "rollouts"
            for index in range(4):
                record_episode(
                    rollout_dir,
                    f"episode-{index}",
                    success=index % 2 == 0,
                    state_offset=index,
                )
            checkpoint = root / "value.pt"
            predicted = root / "predicted.jsonl"
            labeled = root / "labeled.jsonl"

            subprocess.run(
                [
                    sys.executable,
                    "tasks/vla/train_recap_value.py",
                    "--rollout-dir",
                    str(rollout_dir),
                    "--output",
                    str(checkpoint),
                    "--epochs",
                    "2",
                    "--batch-size",
                    "4",
                    "--hidden-size",
                    "8",
                    "--task-embedding-dim",
                    "4",
                    "--num-bins",
                    "21",
                    "--value-min",
                    "-20",
                    "--value-max",
                    "0",
                    "--failure-penalty",
                    "10",
                    "--device",
                    "cpu",
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/recap_predict_values.py",
                    "--rollout-dir",
                    str(rollout_dir),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(predicted),
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/recap_label_rollouts.py",
                    "--input",
                    str(predicted),
                    "--output",
                    str(labeled),
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue(checkpoint.is_file())
            episodes = [json.loads(line) for line in labeled.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(episodes), 4)
            self.assertEqual(sum(len(episode["steps"]) for episode in episodes), 8)
            self.assertTrue(
                all(step["recap_label"] in (-1, 0, 1) for episode in episodes for step in episode["steps"])
            )


if __name__ == "__main__":
    unittest.main()
