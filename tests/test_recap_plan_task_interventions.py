import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.recap_plan_task_interventions import plan_task_interventions


def _manifest(root: Path, task: str, seed: int, success: bool, decisions: int, instruction: str):
    path = root / "attempt-1" / task / f"episode-{seed}" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task": instruction,
                "seed": seed,
                "success": success,
                "metadata": {"task_name": task},
                "steps": [
                    {"decision_index": index, "file": f"steps/{index:06d}.npz"}
                    for index in range(decisions)
                ],
            }
        )
    )


def test_plans_terminal_decisions_and_excludes_perfect_baseline_guards():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        rollout = root / "rollouts"
        _manifest(rollout, "hard", 10, True, 3, "do hard")
        _manifest(rollout, "hard", 11, False, 6, "do hard again")
        _manifest(rollout, "easy", 10, True, 2, "do easy")
        baseline = root / "baseline"
        for task, success in (("hard", 60), ("easy", 100)):
            path = baseline / task / "_result.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "task": task,
                        "success": success,
                        "episodes": 100,
                        "success_rate": success / 100,
                    }
                )
            )

        plan, decisions, instructions = plan_task_interventions(
            rollout,
            baseline_results=baseline,
            minimum_episodes=1,
        )

        assert plan["target_tasks"] == ["hard"]
        assert plan["guard_tasks"] == ["easy"]
        assert decisions == {"schema_version": 1, "tasks": {"hard": 2}}
        assert instructions["tasks"]["hard"] == {
            "10": "do hard",
            "11": "do hard again",
        }
        assert plan["tasks"]["hard"]["candidate_decisions"] == [1, 2, 3]


def test_falls_back_before_step_limit_when_no_reference_succeeds():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _manifest(root, "failed", 1, False, 8, "try")
        _manifest(root, "failed", 2, False, 6, "try")
        plan, decisions, _ = plan_task_interventions(root)
        assert decisions["tasks"]["failed"] == 2
        assert plan["tasks"]["failed"]["decision_source"] == "failure_horizon_fallback"
