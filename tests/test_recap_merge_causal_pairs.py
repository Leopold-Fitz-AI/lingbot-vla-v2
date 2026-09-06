import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from scripts.recap_merge_causal_pairs import merge_causal_pairs


def _member(root: Path, seed: int, pair: int, success: bool, drift: float):
    role = "positive" if success else "negative"
    episode = root / f"{seed}-{pair}-{role}"
    (episode / "steps").mkdir(parents=True)
    np.savez(episode / "steps/000000.npz", executed_action=np.zeros((1, 14)))
    (episode / "manifest.json").write_text(
        json.dumps(
            {
                "episode_id": episode.name,
                "task": "do target",
                "success": success,
                "metadata": {
                    "task_name": "target",
                    "source": "paired_policy_sample",
                    "paired_match_id": f"{seed}:{pair}",
                    "paired_match_role": role,
                    "paired_environment_seed": seed,
                    "paired_original_manifest": f"/raw/{seed}/{pair}/{role}",
                    "paired_image_mae_max": drift,
                },
                "steps": [
                    {
                        "decision_index": 0,
                        "executed_action_length": 1,
                        "file": "steps/000000.npz",
                    }
                ],
            }
        )
    )


def test_merges_complete_pairs_and_caps_each_independent_state():
    with TemporaryDirectory() as directory:
        root = Path(directory) / "source"
        for seed in (1, 2):
            for pair in (0, 1):
                for success in (True, False):
                    _member(root, seed, pair, success, drift=float(pair))
        output = Path(directory) / "merged"
        summary = merge_causal_pairs(
            [root],
            output,
            task_name="target",
            maximum_pairs_per_state=1,
            minimum_states=2,
            minimum_pairs=2,
        )
        assert summary["independent_states"] == 2
        assert summary["pairs"] == 2
        manifests = [json.loads(path.read_text()) for path in output.glob("*/manifest.json")]
        assert len(manifests) == 4
        assert sum(item["success"] for item in manifests) == 2


def test_rejects_insufficient_independent_states():
    with TemporaryDirectory() as directory:
        root = Path(directory) / "source"
        for success in (True, False):
            _member(root, 1, 0, success, drift=0)
        with pytest.raises(ValueError, match="independent causal states"):
            merge_causal_pairs(
                [root],
                Path(directory) / "merged",
                task_name="target",
                minimum_states=2,
            )
