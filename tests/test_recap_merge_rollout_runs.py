import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deploy.recap_rollout_recorder import RecapEpisodeRecorder
from lingbotvla.recap.rollouts import load_rollout_decisions
from scripts.recap_merge_rollout_runs import merge_rollout_runs


def _record_episode(root: Path, branch: str, episode_id: str, *, success: bool) -> None:
    recorder = RecapEpisodeRecorder(
        root / branch,
        episode_id=episode_id,
        task="click the bell",
        seed=7,
    )
    recorder.record_step(
        {"observation.state": np.zeros(14, dtype=np.float32), "task": "click the bell"},
        np.zeros((5, 14), dtype=np.float32),
        np.zeros((2, 14), dtype=np.float32),
    )
    recorder.record_step(
        {"observation.state": np.ones(14, dtype=np.float32), "task": "click the bell"},
        np.zeros((5, 14), dtype=np.float32),
        np.zeros((1, 14), dtype=np.float32),
        terminated=success,
        truncated=not success,
    )
    recorder.finalize(success=success, terminal_reason="success" if success else "limit")


class MergeRolloutRunsTest(unittest.TestCase):
    def test_merges_duplicate_episode_ids_across_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_a = root / "run_a"
            run_b = root / "run_b"
            for run in (run_a, run_b):
                _record_episode(run, "positive", "click_bell-episode-0", success=True)
                _record_episode(run, "null", "click_bell-episode-0", success=False)

            output = root / "merged"
            summary = merge_rollout_runs([run_a, run_b], output)

            self.assertEqual(summary["episodes"], 4)
            self.assertEqual(summary["decisions"], 8)
            decisions = load_rollout_decisions(output)
            self.assertEqual(len(decisions), 8)
            episode_ids = {decision.episode_id for decision in decisions}
            self.assertEqual(len(episode_ids), 4)
            self.assertTrue(all("run_" in episode_id for episode_id in episode_ids))
            for manifest_path in output.rglob("manifest.json"):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["episode_id"], manifest_path.parent.name)
                self.assertIn("merged_from", manifest["metadata"])

    def test_fails_closed_without_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_a = root / "run_a"
            _record_episode(run_a, "positive", "click_bell-episode-0", success=True)
            broken = run_a / "positive" / "broken-episode"
            (broken / "steps").mkdir(parents=True)
            (broken / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "episode_id": "broken-episode",
                        "task": "click the bell",
                        "success": False,
                        "steps": [
                            {
                                "decision_index": 0,
                                "file": "steps/000000.npz",
                                "executed_action_length": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "merged"
            with self.assertRaises(FileNotFoundError):
                merge_rollout_runs([run_a], output)
            self.assertFalse(output.exists())

    def test_rejects_existing_output_and_missing_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_a = root / "run_a"
            _record_episode(run_a, "positive", "click_bell-episode-0", success=True)
            with self.assertRaises(FileNotFoundError):
                merge_rollout_runs([root / "missing"], root / "merged")
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                merge_rollout_runs([run_a], existing)


if __name__ == "__main__":
    unittest.main()
