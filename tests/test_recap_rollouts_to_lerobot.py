import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.recap_rollouts_to_lerobot import (
    load_labeled_decisions,
    pad_action_chunk,
    prepare_rgb_image,
)


class ActionChunkPaddingTest(unittest.TestCase):
    def test_pads_with_final_action_and_marks_padding(self):
        action = np.arange(42, dtype=np.float32).reshape(3, 14)
        padded, is_pad = pad_action_chunk(action, 5)
        self.assertEqual(padded.shape, (5, 14))
        self.assertEqual(is_pad.tolist(), [False, False, False, True, True])
        np.testing.assert_array_equal(padded[:3], action)
        np.testing.assert_array_equal(padded[3], action[-1])
        np.testing.assert_array_equal(padded[4], action[-1])

    def test_rejects_wrong_action_dimension(self):
        with self.assertRaisesRegex(ValueError, r"\[T,14\]"):
            pad_action_chunk(np.zeros((2, 13), dtype=np.float32), 50)

    def test_rejects_oversized_chunk(self):
        with self.assertRaisesRegex(ValueError, "length"):
            pad_action_chunk(np.zeros((51, 14), dtype=np.float32), 50)


class RgbImagePreparationTest(unittest.TestCase):
    def test_resizes_rgb_to_dataset_shape(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        output, resized = prepare_rgb_image(image)
        self.assertTrue(resized)
        self.assertEqual(output.shape, (240, 320, 3))
        self.assertEqual(output.dtype, np.uint8)

    def test_keeps_matching_image_without_copy_requirement(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        output, resized = prepare_rgb_image(image)
        self.assertFalse(resized)
        np.testing.assert_array_equal(output, image)


class LabeledDecisionLoadTest(unittest.TestCase):
    def test_loads_decision_annotations(self):
        payload = {
            "episode_id": "episode-a",
            "steps": [
                {
                    "decision_index": 2,
                    "recap_label": 0,
                    "recap_advantage": -3.5,
                    "recap_return": -10.0,
                    "value": -6.5,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            labels, digest = load_labeled_decisions(path)
        self.assertEqual(len(digest), 64)
        self.assertEqual(labels[("episode-a", 2)]["recap_label"], 0)
        self.assertEqual(labels[("episode-a", 2)]["recap_advantage"], -3.5)

    def test_rejects_duplicate_decisions(self):
        step = {
            "decision_index": 0,
            "recap_label": 1,
            "recap_advantage": 1.0,
            "recap_return": 0.0,
            "value": -1.0,
        }
        payload = {"episode_id": "episode-a", "steps": [step, step]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_labeled_decisions(path)


if __name__ == "__main__":
    unittest.main()
