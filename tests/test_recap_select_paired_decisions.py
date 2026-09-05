import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.recap_select_paired_decisions import select_paired_decisions


def _write_episode(
    root: Path,
    *,
    source: str,
    seed: int,
    success: bool,
    action: float,
    observation=1.0,
    task_name="click_bell",
):
    episode = root / source / f"{task_name}-episode-{seed}"
    (episode / "steps").mkdir(parents=True)
    np.savez_compressed(
        episode / "steps/000000.npz",
        **{
            "observation::observation.state": np.array([observation], np.float32),
            "observation::observation.images.cam_high": np.zeros((2, 2, 3), np.uint8),
            "executed_action": np.full((2, 1), action, np.float32),
            "generated_action": np.full((2, 1), action, np.float32),
        },
    )
    manifest = {
        "schema_version": 1,
        "episode_id": f"episode-{seed}",
        "task": task_name.replace("_", " "),
        "success": success,
        "metadata": {"task_name": task_name},
        "seed": seed,
        "num_decisions": 1,
        "steps": [
            {
                "decision_index": 0,
                "executed_action_length": 2,
                "file": "steps/000000.npz",
                "terminated": False,
            }
        ],
    }
    (episode / "manifest.json").write_text(json.dumps(manifest))
    return root / source


def test_selects_only_mixed_groups_and_verifies_pairing():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        left = _write_episode(root, source="left", seed=1, success=True, action=1)
        right = _write_episode(root, source="right", seed=1, success=False, action=2)
        # An all-success seed must not enter the selected dataset.
        _write_episode(root, source="left", seed=2, success=True, action=3)
        _write_episode(root, source="right", seed=2, success=True, action=4)
        output = root / "selected"
        summary = select_paired_decisions([left, right], output)
        assert summary["mixed_outcome_seeds"] == 1
        assert summary["episodes"] == 2
        assert summary["positive"] == 1
        assert summary["negative"] == 1
        assert len(list(output.glob("*/manifest.json"))) == 2
        assert summary["balance_outcomes"] is False


def test_task_filter_prevents_cross_task_seed_collisions():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        left = _write_episode(root, source="left", seed=1, success=True, action=1)
        right = _write_episode(root, source="right", seed=1, success=False, action=2)
        _write_episode(
            root,
            source="left",
            seed=1,
            success=False,
            action=3,
            task_name="other_task",
        )
        _write_episode(
            root,
            source="right",
            seed=1,
            success=True,
            action=4,
            task_name="other_task",
        )
        summary = select_paired_decisions(
            [left, right],
            root / "selected",
            task_name="click_bell",
        )
        assert summary["task_name"] == "click_bell"
        assert summary["episodes"] == 2


def test_rejects_nonidentical_observations_within_state_group():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        left = _write_episode(
            root, source="left", seed=1, success=True, action=1, observation=1
        )
        right = _write_episode(
            root, source="right", seed=1, success=False, action=2, observation=2
        )
        try:
            select_paired_decisions([left, right], root / "selected")
        except ValueError as exc:
            assert "exceed pairing tolerance" in str(exc)
        else:
            raise AssertionError("Expected observation mismatch to be rejected")
