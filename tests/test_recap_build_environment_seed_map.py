import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.recap_build_environment_seed_map import build_seed_map


def _manifest(root: Path, task: str, seed: int):
    path = root / task / str(seed) / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "success": True,
                "metadata": {"task_name": task},
            }
        )
    )


def test_freezes_sorted_intersection_across_sources():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for seed in (3, 1, 2):
            _manifest(root / "a", "task", seed)
        for seed in (4, 3, 2):
            _manifest(root / "b", "task", seed)
        result = build_seed_map([root / "a", root / "b"], seeds_per_task=2)
        assert result["tasks"] == {"task": [2, 3]}
