import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import scripts.recap_open_loop_check as check_script
from deploy.recap_rollout_recorder import RecapEpisodeRecorder


IMAGE_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]

ACTION_SCALE = 2.0


def record_episode(root, episode_id, *, task, action_sign, num_decisions=2):
    """Record a bundle whose executed actions encode the decision's label."""

    recorder = RecapEpisodeRecorder(
        root,
        episode_id=episode_id,
        task=task,
        metadata={"task_name": task.replace(" ", "_")},
    )
    for decision_index in range(num_decisions):
        executed = np.full((4, 14), action_sign * ACTION_SCALE, dtype=np.float32)
        observation = {
            "observation.state": np.zeros(14, dtype=np.float32),
        }
        for camera_index, key in enumerate(IMAGE_KEYS):
            observation[key] = np.full(
                (6, 10, 3), fill_value=10 * camera_index, dtype=np.uint8
            )
        recorder.record_step(
            observation,
            np.zeros((4, 14), dtype=np.float32),
            executed,
            terminated=decision_index == num_decisions - 1,
        )
    recorder.finalize(success=True, terminal_reason="success")


def write_labels(path, entries):
    """entries: list of (episode_id, [(decision_index, recap_label), ...], task)."""

    with open(path, "w", encoding="utf-8") as file:
        for episode_id, steps, task in entries:
            json.dump(
                {
                    "episode_id": episode_id,
                    "task": task,
                    "success": True,
                    "steps": [
                        {"decision_index": index, "recap_label": label, "value": -1.0}
                        for index, label in steps
                    ],
                },
                file,
            )
            file.write("\n")


class FakeFeatureTransform:
    """Stand-in with a known /2 normalization inverted by unapply."""

    org_features = {
        "images": list(IMAGE_KEYS),
        "states": ["observation.state"],
        "actions": ["action"],
    }
    recap_enabled = False

    def apply(self, item, policy_eval=False):
        images = torch.stack([item[key] for key in self.org_features["images"]], dim=0)
        lang_tokens = torch.arange(1, len(str(item["task"])) + 1, dtype=torch.long)
        return {
            "images": images,
            "img_masks": torch.ones(images.shape[0], dtype=torch.bool),
            "state": torch.as_tensor(item["observation.state"], dtype=torch.float32),
            "lang_tokens": lang_tokens,
            "lang_masks": torch.ones(lang_tokens.shape[0], dtype=torch.bool),
            "image_grid_thw": torch.tensor(
                [[1, images.shape[-2], images.shape[-1]]] * images.shape[0],
                dtype=torch.long,
            ),
        }

    def unapply(self, item):
        return {"action": item["actions"] * 2.0}


class FakeAdapterFlow:
    """Condition-steerable double: positive -> +1, null -> 0, negative -> -1.

    Actions are returned in normalized space (unapply multiplies by 2), so raw
    actions land on +/-ACTION_SCALE. A small flow-noise term verifies paired
    seeding: identical seeds must give identical noise across conditions.
    """

    def __init__(self, chunk=6, action_dim=14):
        self.chunk = chunk
        self.action_dim = action_dim
        self.calls = []

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        image_grid_thw=None,
        recap_condition_id=None,
        **kwargs,
    ):
        condition = int(torch.as_tensor(recap_condition_id).reshape(-1)[0].item())
        sign = {1: 1.0, -1: 0.0, 0: -1.0}[condition]
        flow_noise = torch.randn((), device=state.device)
        self.calls.append({"condition": condition, "noise": float(flow_noise)})
        value = sign * ACTION_SCALE / 2.0 + 0.01 * flow_noise
        return torch.full(
            (1, self.chunk, self.action_dim),
            float(value),
            device=state.device,
            dtype=state.dtype,
        )


def fake_load_frozen_flow_model(flow):
    def loader(checkpoint_path, robot_config, **kwargs):
        return flow, FakeFeatureTransform(), {"image_size": 8}

    return loader


def build_rollout_set(root):
    rollout_dir = root / "rollouts"
    record_episode(rollout_dir, "pos-a", task="pick cup", action_sign=1.0)
    record_episode(rollout_dir, "pos-b", task="open drawer", action_sign=1.0)
    record_episode(rollout_dir, "neg-a", task="pick cup", action_sign=-1.0)
    record_episode(rollout_dir, "neg-b", task="stack bowls", action_sign=-1.0)
    labels = root / "labels.jsonl"
    write_labels(
        labels,
        [
            ("pos-a", [(0, 1), (1, 1)], "pick cup"),
            ("pos-b", [(0, 1), (1, 1)], "open drawer"),
            ("neg-a", [(0, 0), (1, 0)], "pick cup"),
            ("neg-b", [(0, 0), (1, 0)], "stack bowls"),
        ],
    )
    return rollout_dir, labels


class LabelMapTest(unittest.TestCase):
    def test_parses_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.jsonl"
            write_labels(labels, [("ep", [(0, 1), (1, -1), (2, 0)], "task")])
            mapping = check_script.load_label_map(labels)
            self.assertEqual(
                mapping,
                {("ep", 0): 1, ("ep", 1): -1, ("ep", 2): 0},
            )

    def test_malformed_line_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.jsonl"
            labels.write_text('{"episode_id": "ep", "steps": [{"decision_index": 0}]}\n')
            with self.assertRaisesRegex(ValueError, "recap_label"):
                check_script.load_label_map(labels)

    def test_invalid_label_value_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.jsonl"
            write_labels(labels, [("ep", [(0, 7)], "task")])
            with self.assertRaisesRegex(ValueError, "recap_label"):
                check_script.load_label_map(labels)


class SelectionTest(unittest.TestCase):
    def test_round_robin_stratifies_across_tasks_and_fails_closed_on_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout_dir, labels = build_rollout_set(root)
            decisions = check_script.load_rollout_decisions(rollout_dir)
            mapping = check_script.load_label_map(labels)
            selected = check_script.select_labeled_decisions(decisions, mapping, 2)
            self.assertEqual(len(selected["positive"]), 2)
            self.assertEqual(len(selected["negative"]), 2)
            # Round-robin picks one decision per task before repeating a task.
            positive_tasks = [item.task_name for item in selected["positive"]]
            self.assertEqual(positive_tasks, ["open_drawer", "pick_cup"])

            mapping.pop(("pos-a", 0))
            with self.assertRaisesRegex(KeyError, "pos-a::0"):
                check_script.select_labeled_decisions(decisions, mapping, 2)


class AggregationTest(unittest.TestCase):
    def test_ordering_detection(self):
        healthy = {
            "positive": [
                {"errors": {"positive": 0.1, "null": 0.5, "negative": 0.9},
                 "residual_positive_null": 1.0},
                {"errors": {"positive": 0.2, "null": 0.4, "negative": 0.8},
                 "residual_positive_null": 1.0},
            ],
            "negative": [
                {"errors": {"negative": 0.1, "null": 0.5, "positive": 0.9},
                 "residual_positive_null": 1.0},
            ],
        }
        summary = check_script.aggregate_results(healthy)
        self.assertTrue(summary["ordering_pass"])
        self.assertEqual(
            summary["per_class"]["positive"]["matching_condition_win_rate_vs_null"], 1.0
        )
        self.assertAlmostEqual(
            summary["residual_magnitude_positive_minus_null"], 1.0
        )

        flipped = {
            "positive": [
                {"errors": {"positive": 0.9, "null": 0.5, "negative": 0.1},
                 "residual_positive_null": 1.0},
            ],
            "negative": [
                {"errors": {"negative": 0.9, "null": 0.5, "positive": 0.1},
                 "residual_positive_null": 1.0},
            ],
        }
        summary = check_script.aggregate_results(flipped)
        self.assertFalse(summary["ordering_pass"])
        self.assertFalse(summary["per_class"]["positive"]["ordering_pass"])
        self.assertFalse(summary["per_class"]["negative"]["ordering_pass"])

    def test_win_rate_counts_only_strict_wins_over_null(self):
        rows = {
            "positive": [
                {"errors": {"positive": 0.4, "null": 0.5, "negative": 0.9},
                 "residual_positive_null": 1.0},
                {"errors": {"positive": 0.6, "null": 0.5, "negative": 0.9},
                 "residual_positive_null": 1.0},
            ],
            "negative": [
                {"errors": {"negative": 0.4, "null": 0.5, "positive": 0.9},
                 "residual_positive_null": 1.0},
            ],
        }
        summary = check_script.aggregate_results(rows)
        self.assertEqual(
            summary["per_class"]["positive"]["matching_condition_win_rate_vs_null"], 0.5
        )


class EndToEndTest(unittest.TestCase):
    def test_main_paired_conditions_and_ordering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout_dir, labels = build_rollout_set(root)
            flow = FakeAdapterFlow()
            with mock.patch.object(
                check_script, "load_frozen_flow_model", fake_load_frozen_flow_model(flow)
            ):
                summary = check_script.main(
                    [
                        "--checkpoint",
                        "/fake/checkpoint",
                        "--rollout-dir",
                        str(rollout_dir),
                        "--labels",
                        str(labels),
                        "--samples-per-class",
                        "2",
                        "--compare-steps",
                        "3",
                        "--device",
                        "cpu",
                        "--seed",
                        "7",
                    ]
                )
            self.assertTrue(summary["ordering_pass"])
            for class_name, matching in (("positive", "positive"), ("negative", "negative")):
                errors = summary["per_class"][class_name]["mean_error"]
                self.assertLess(errors[matching], errors["null"])
                other = "negative" if matching == "positive" else "positive"
                self.assertLess(errors["null"], errors[other])
                self.assertEqual(
                    summary["per_class"][class_name]["matching_condition_win_rate_vs_null"],
                    1.0,
                )
            # Raw residual is |+scale - 0| = ACTION_SCALE after unapply.
            self.assertAlmostEqual(
                summary["residual_magnitude_positive_minus_null"],
                ACTION_SCALE * np.sqrt(14),
                places=4,
            )

            # Paired seeding: each decision's three conditions share one noise
            # draw, and consecutive decisions use different seeds.
            self.assertEqual(len(flow.calls), 12)
            for decision_index in range(4):
                triple = flow.calls[3 * decision_index : 3 * decision_index + 3]
                self.assertEqual(
                    {call["condition"] for call in triple}, {1, -1, 0}
                )
                noises = {call["noise"] for call in triple}
                self.assertEqual(len(noises), 1)
            all_noises = [call["noise"] for call in flow.calls[::3]]
            self.assertEqual(len(set(all_noises)), 4)

    def test_missing_labels_fail_closed_with_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout_dir, labels = build_rollout_set(root)
            # Drop every step of one episode from the labels file.
            lines = labels.read_text(encoding="utf-8").splitlines()
            labels.write_text(
                "\n".join(
                    line for line in lines if '"neg-b"' not in line
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                check_script,
                "load_frozen_flow_model",
                fake_load_frozen_flow_model(FakeAdapterFlow()),
            ):
                with self.assertRaisesRegex(KeyError, "neg-b::0"):
                    check_script.main(
                        [
                            "--checkpoint",
                            "/fake/checkpoint",
                            "--rollout-dir",
                            str(rollout_dir),
                            "--labels",
                            str(labels),
                            "--device",
                            "cpu",
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
