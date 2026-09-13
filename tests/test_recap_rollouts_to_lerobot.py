import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.recap_rollouts_to_lerobot import (
    _flattened_stats,
    expected_matched_keys,
    load_labeled_decisions,
    pad_action_chunk,
    parse_keep_labels,
    prepare_rgb_image,
)


class FlattenedStatsTest(unittest.TestCase):
    def test_flattens_chunk_feature_to_per_dimension_stats(self):
        array = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        stats = _flattened_stats(array)
        flat = array.reshape(-1, 4)
        self.assertEqual(stats["mean"].shape, (4,))
        np.testing.assert_allclose(stats["mean"], flat.mean(axis=0))
        np.testing.assert_allclose(stats["std"], flat.std(axis=0))
        np.testing.assert_array_equal(stats["count"], np.array([6]))
        self.assertEqual(stats["q50"].shape, (4,))


class KeepLabelsFilterTest(unittest.TestCase):
    def test_none_keeps_everything(self):
        self.assertIsNone(parse_keep_labels(None))

    def test_parses_positive_only(self):
        self.assertEqual(parse_keep_labels("1"), frozenset({1}))

    def test_parses_multiple_labels(self):
        self.assertEqual(parse_keep_labels("1, 0"), frozenset({0, 1}))

    def test_rejects_unknown_labels(self):
        with self.assertRaisesRegex(ValueError, "keep-labels"):
            parse_keep_labels("1,2")

    def test_rejects_empty_selection(self):
        with self.assertRaisesRegex(ValueError, "keep-labels"):
            parse_keep_labels(" , ")

    def test_expected_keys_only_count_selected_labels(self):
        labels = {
            ("ep-a", 0): {"recap_label": 1},
            ("ep-a", 1): {"recap_label": 0},
            ("ep-b", 0): {"recap_label": -1},
            ("ep-b", 1): {"recap_label": 1},
        }
        self.assertEqual(
            expected_matched_keys(labels, None),
            {("ep-a", 0), ("ep-a", 1), ("ep-b", 0), ("ep-b", 1)},
        )
        self.assertEqual(
            expected_matched_keys(labels, frozenset({1})),
            {("ep-a", 0), ("ep-b", 1)},
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
