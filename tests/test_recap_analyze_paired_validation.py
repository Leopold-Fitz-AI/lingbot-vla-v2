import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.recap_analyze_paired_validation import analyze


def _manifest(root: Path, condition: str, task: str, seed: int, success: bool):
    path = root / f"{condition}_s40" / task / f"episode-{seed}" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "task": f"do {task} seed {seed}",
                "seed": seed,
                "success": success,
                "metadata": {"task_name": task, "recap_condition": condition},
            }
        )
    )


def test_analyzes_common_seeds_and_requires_complete_strict_order():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        outcomes = {
            "positive": {1: True, 2: True},
            "null": {1: True, 2: False},
            "negative": {1: False, 2: False},
        }
        for condition, values in outcomes.items():
            for seed, success in values.items():
                _manifest(root, condition, "promote", seed, success)
        for condition in ("positive", "null"):
            _manifest(root, condition, "incomplete", 1, condition == "positive")
            _manifest(root, condition, "incomplete", 2, False)
        _manifest(root, "negative", "incomplete", 1, False)

        result = analyze(root, [40], expected_per_task=2)
        assert result["promoted_tasks"] == ["promote"]
        assert result["tasks"]["promote"]["paired_success"] == {
            "positive": 2,
            "null": 1,
            "negative": 0,
        }
        assert result["tasks"]["incomplete"]["paired_common_episodes"] == 1
        assert not result["tasks"]["incomplete"]["promoted"]
