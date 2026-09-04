import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deploy.recap_rollout_recorder import RecapEpisodeRecorder


class RecapEpisodeRecorderTest(unittest.TestCase):
    def test_records_exact_executed_action_and_atomic_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RecapEpisodeRecorder(
                directory,
                episode_id="pick cup/seed 1",
                task="pick cup",
                seed=1,
                metadata={"policy": "test"},
            )
            generated = np.arange(20, dtype=np.float32).reshape(5, 4)
            executed = generated[:2]
            step_path = recorder.record_step(
                {
                    "observation.state": np.arange(4, dtype=np.float32),
                    "observation.images.camera_top": np.zeros((4, 4, 3), dtype=np.uint8),
                    "task": "pick cup",
                },
                generated,
                executed,
                env_step_before=10,
                env_step_after=12,
                terminated=True,
                reward=0,
                info={"latency_ms": np.float32(12.5)},
            )
            manifest_path = recorder.finalize(
                success=True,
                terminal_reason="success",
            )

            self.assertTrue(step_path.is_file())
            self.assertTrue(manifest_path.is_file())
            arrays = np.load(step_path)
            self.assertTrue(np.array_equal(arrays["generated_action"], generated))
            self.assertTrue(np.array_equal(arrays["executed_action"], executed))
            self.assertIn("observation::observation.state", arrays.files)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertTrue(manifest["success"])
            self.assertEqual(manifest["steps"][0]["executed_action_length"], 2)
            self.assertEqual(manifest["steps"][0]["env_step_after"], 12)
            self.assertEqual(manifest["steps"][0]["info"]["latency_ms"], 12.5)
            self.assertNotIn(".tmp", {path.suffix for path in Path(directory).rglob("*")})

    def test_rejects_duplicate_episode_and_append_after_finalize(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RecapEpisodeRecorder(
                directory,
                episode_id="episode-1",
                task="pick cup",
            )
            recorder.finalize(success=False, terminal_reason="timeout")
            with self.assertRaisesRegex(RuntimeError, "finalized"):
                recorder.record_step({}, np.zeros(2), np.zeros(2))
            with self.assertRaises(FileExistsError):
                RecapEpisodeRecorder(
                    directory,
                    episode_id="episode-1",
                    task="pick cup",
                )


if __name__ == "__main__":
    unittest.main()
