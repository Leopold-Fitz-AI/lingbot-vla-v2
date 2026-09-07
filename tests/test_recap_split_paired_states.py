import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from scripts.recap_split_paired_states import split_paired_states


def _pair(root: Path, seed: int, pair: int):
    for success in (False, True):
        episode = root / f"{seed}-{pair}-{int(success)}"
        (episode / "steps").mkdir(parents=True)
        np.savez(episode / "steps" / "000000.npz", action=np.zeros((1, 14)))
        row = {
            "episode_id": episode.name,
            "task": "instruction",
            "success": success,
            "metadata": {
                "task_name": "task",
                "paired_environment_seed": seed,
                "paired_match_id": f"pair-{pair}",
            },
            "steps": [{"file": "steps/000000.npz", "executed_action_length": 1}],
        }
        (episode / "manifest.json").write_text(json.dumps(row))


def test_split_keeps_states_and_pairs_disjoint():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for seed in range(12):
            _pair(root / "source", seed, 0)
            _pair(root / "source", seed, 1)
        result = split_paired_states(
            root / "source",
            root / "split",
            task_name="task",
            minimum_train_states=8,
            minimum_holdout_states=2,
        )
        assert result["train_state_count"] == 9
        assert result["holdout_state_count"] == 3
        assert set(result["train_states"]).isdisjoint(result["holdout_states"])
        assert len(list((root / "split" / "train").rglob("manifest.json"))) == 36
        assert len(list((root / "split" / "holdout").rglob("manifest.json"))) == 12


def test_rejects_too_few_independent_states():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for seed in range(8):
            _pair(root / "source", seed, 0)
        with pytest.raises(ValueError, match="cannot provide"):
            split_paired_states(
                root / "source", root / "split", task_name="task"
            )
