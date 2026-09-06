import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from scripts.recap_merge_causal_positives import merge_causal_positives


def _paired(root: Path, task: str, index: int, *, success: bool):
    episode = root / f"episode-{index}"
    (episode / "steps").mkdir(parents=True)
    np.savez(episode / "steps/000000.npz", executed_action=np.zeros((1, 14)))
    (episode / "manifest.json").write_text(
        json.dumps(
            {
                "episode_id": f"source-{index}",
                "task": f"do {task}",
                "success": success,
                "num_decisions": 1,
                "metadata": {
                    "task_name": task,
                    "source": "paired_policy_sample",
                    "paired_match_role": "positive" if success else "negative",
                    "paired_original_manifest": f"/raw/{task}/{index}/manifest.json",
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


def test_merges_only_positive_members_for_requested_task():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        first, second = root / "first", root / "second"
        _paired(first, "target", 1, success=True)
        _paired(first, "target", 2, success=False)
        _paired(second, "target", 3, success=True)
        _paired(second, "other", 4, success=True)
        output = root / "merged"
        summary = merge_causal_positives(
            [first, second], output, task_name="target", minimum_positive=2
        )
        assert summary["positive_episodes"] == 2
        manifests = [json.loads(path.read_text()) for path in output.glob("*/manifest.json")]
        assert len(manifests) == 2
        assert all(item["success"] for item in manifests)
        assert len((output / "episodes.jsonl").read_text().splitlines()) == 2


def test_rejects_duplicate_underlying_policy_sample():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        first, second = root / "first", root / "second"
        _paired(first, "target", 1, success=True)
        _paired(second, "target", 1, success=True)
        with pytest.raises(ValueError, match="Duplicate paired source"):
            merge_causal_positives(
                [first, second], root / "merged", task_name="target"
            )
