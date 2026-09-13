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
    group_decisions_by_episode,
    load_decision_images,
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
            {
                "observation.state": np.array([state_offset, decision_index], dtype=np.float32),
                "observation.images.cam_high": np.full(
                    (4, 6, 3),
                    fill_value=10 + decision_index,
                    dtype=np.uint8,
                ),
                "observation.images.cam_left_wrist": np.full(
                    (4, 6, 3),
                    fill_value=20 + decision_index,
                    dtype=np.uint8,
                ),
            },
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

    def test_group_decisions_keeps_episode_order(self):
        with tempfile.TemporaryDirectory() as directory:
            record_episode(directory, "success", success=True)
            record_episode(directory, "failure", success=False, state_offset=1)
            decisions = load_rollout_decisions(directory, failure_penalty=10)
            grouped = group_decisions_by_episode(decisions)
            self.assertEqual(set(grouped), {"success", "failure"})
            self.assertEqual([item.decision_index for item in grouped["success"]], [0, 1])

    def test_camera_images_stack_in_channel_first_float_form(self):
        with tempfile.TemporaryDirectory() as directory:
            record_episode(directory, "success", success=True)
            decisions = load_rollout_decisions(directory, failure_penalty=10)
            images = load_decision_images(decisions[1], image_size=4)
            self.assertEqual(tuple(images.shape), (2, 3, 4, 4))
            self.assertEqual(images.dtype, np.float32)
            self.assertGreater(images.min(), 0.0)
            self.assertLessEqual(images.max(), 1.0)
            self.assertGreater(images[1].mean(), images[0].mean())

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

    def test_label_rollouts_from_directory_with_checkpoint(self):
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
                    "1",
                    "--batch-size",
                    "4",
                    "--hidden-size",
                    "8",
                    "--task-embedding-dim",
                    "4",
                    "--num-bins",
                    "11",
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
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/recap_label_rollouts.py",
                    "--input",
                    str(rollout_dir),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(labeled),
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            episodes = [json.loads(line) for line in labeled.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(episodes), 4)
            self.assertEqual(sum(len(episode["steps"]) for episode in episodes), 8)
            self.assertTrue(
                all("observation_ref" in step for episode in episodes for step in episode["steps"])
            )
            self.assertTrue(
                all(step["recap_label"] in (-1, 0, 1) for episode in episodes for step in episode["steps"])
            )
            self.assertIn("episodes", completed.stdout)

    def test_label_rollouts_from_directory_outcome_mode_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout_dir = root / "rollouts"
            record_episode(rollout_dir, "success", success=True)
            record_episode(rollout_dir, "failure", success=False, state_offset=1)
            labeled = root / "labeled.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/recap_label_rollouts.py",
                    "--input",
                    str(rollout_dir),
                    "--output",
                    str(labeled),
                    "--label-mode",
                    "outcome",
                    "--failure-penalty",
                    "10",
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            episodes = {
                episode["episode_id"]: episode
                for episode in (json.loads(line) for line in labeled.read_text(encoding="utf-8").splitlines())
            }
            self.assertEqual(
                [step["recap_label"] for step in episodes["success"]["steps"]],
                [1, 1],
            )
            self.assertEqual(
                [step["recap_label"] for step in episodes["failure"]["steps"]],
                [0, 0],
            )

    def test_directory_advantage_mode_requires_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout_dir = root / "rollouts"
            record_episode(rollout_dir, "success", success=True)
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/recap_label_rollouts.py",
                    "--input",
                    str(rollout_dir),
                    "--output",
                    str(root / "labeled.jsonl"),
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("--checkpoint", completed.stderr)

    def test_visual_encoder_train_predict_and_n_step_label(self):
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
            checkpoint = root / "visual_value.pt"
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
                    "--encoder",
                    "visual_task",
                    "--image-size",
                    "8",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--hidden-size",
                    "8",
                    "--task-embedding-dim",
                    "4",
                    "--num-bins",
                    "11",
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
                    "--n-step",
                    "1",
                    "--positive-fraction",
                    "0.5",
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(checkpoint, map_location="cpu")
            self.assertEqual(payload["model_type"], "visual_task_categorical_value")
            episodes = [json.loads(line) for line in labeled.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(episodes), 4)
            self.assertTrue(
                all(step["recap_label"] in (-1, 0, 1) for episode in episodes for step in episode["steps"])
            )

    def test_recap_visual_value_script_locks_visual_encoder(self):
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
            checkpoint = root / "visual_value.pt"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/recap_visual_value.py",
                    "--rollout-dir",
                    str(rollout_dir),
                    "--output",
                    str(checkpoint),
                    "--image-size",
                    "8",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--hidden-size",
                    "8",
                    "--task-embedding-dim",
                    "4",
                    "--num-bins",
                    "11",
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
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(checkpoint, map_location="cpu")
            self.assertEqual(payload["model_type"], "visual_task_categorical_value")
            rejected = subprocess.run(
                [
                    sys.executable,
                    "scripts/recap_visual_value.py",
                    "--encoder",
                    "state_task",
                    "--rollout-dir",
                    str(rollout_dir),
                    "--output",
                    str(root / "ignored.pt"),
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertTrue("encoder" in rejected.stderr.lower() or "unrecognized" in rejected.stderr.lower())

    def test_vlm_pooled_train_predict_from_embedding_cache(self):
        from lingbotvla.recap.rollouts import load_rollout_decisions
        from lingbotvla.recap.vlm_pool import write_vlm_embedding_cache

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
            decisions = load_rollout_decisions(rollout_dir, failure_penalty=10)
            embeddings = torch.stack(
                [
                    torch.tensor(
                        [float(item.empirical_return), float(item.decision_index)],
                        dtype=torch.float32,
                    )
                    for item in decisions
                ]
            )
            cache_path = root / "vlm_cache.pt"
            write_vlm_embedding_cache(
                cache_path,
                embeddings=embeddings,
                episode_ids=[item.episode_id for item in decisions],
                decision_indices=[item.decision_index for item in decisions],
            )
            checkpoint = root / "vlm_value.pt"
            predicted = root / "predicted.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "tasks/vla/train_recap_value.py",
                    "--rollout-dir",
                    str(rollout_dir),
                    "--output",
                    str(checkpoint),
                    "--encoder",
                    "vlm_pooled",
                    "--embedding-cache",
                    str(cache_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--num-bins",
                    "11",
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
                    "--embedding-cache",
                    str(cache_path),
                    "--output",
                    str(predicted),
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(checkpoint, map_location="cpu")
            self.assertEqual(payload["model_type"], "vlm_pooled_categorical_value")
            self.assertEqual(len(predicted.read_text(encoding="utf-8").splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
