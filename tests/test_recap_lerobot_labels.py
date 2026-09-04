import json
import tempfile
import unittest
from pathlib import Path

from scripts.recap_attach_lerobot_labels import load_label_map


class LeRobotLabelMapTest(unittest.TestCase):
    def test_reads_explicit_episode_and_frame_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "episode_index": 3,
                        "steps": [
                            {"frame_index": 7, "recap_label": 1},
                            {"frame_index": 8, "recap_label": 0},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_label_map(path), {(3, 7): 1, (3, 8): 0})

    def test_decision_index_fallback_must_be_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "episode_id": 4,
                        "steps": [{"decision_index": 2, "recap_label": -1}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "frame_index"):
                load_label_map(path)
            self.assertEqual(
                load_label_map(path, use_decision_index=True),
                {(4, 2): -1},
            )

    def test_rejects_duplicate_and_invalid_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            duplicate = {
                "episode_index": 0,
                "steps": [
                    {"frame_index": 1, "recap_label": 1},
                    {"frame_index": 1, "recap_label": 0},
                ],
            }
            path.write_text(json.dumps(duplicate) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_label_map(path)

            duplicate["steps"] = [{"frame_index": 1, "recap_label": 2}]
            path.write_text(json.dumps(duplicate) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid"):
                load_label_map(path)


if __name__ == "__main__":
    unittest.main()
