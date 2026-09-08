#!/usr/bin/env python3
"""Outcome-blind progress view; never aggregates partial final success rates."""

import argparse
import json
from pathlib import Path


def study_status(root):
    status = json.loads((root / "status.json").read_text())
    status["final_seeds_opened"] = (root / "final_opened.json").exists()
    stages = {}
    for job in sorted((root / "evaluations").glob("*/*/*/*/job.json")):
        spec = json.loads(job.read_text())
        phase = stages.setdefault(spec["phase"], {"planned_in_created_jobs": 0, "validated_episodes": 0,
                                                  "completed_jobs": 0, "incomplete_jobs": 0})
        phase["planned_in_created_jobs"] += len(spec["tasks"]) * spec["count"]
        done = job.parent / "done.json"
        if done.exists():
            phase["validated_episodes"] += len(json.loads(done.read_text())["manifests"])
            phase["completed_jobs"] += 1
        else:
            phase["incomplete_jobs"] += 1
    status["stages"] = stages
    status["note"] = "No interim outcome aggregation. Incomplete jobs may be running or require infrastructure recovery."
    result = root / "final_result.json"
    if result.exists():
        final = json.loads(result.read_text())
        status["final"] = {key: final[key] for key in ("passed", "macro_delta", "two_sided_exact_p",
                                                     "paired_bootstrap_95ci", "registry_tasks", "scope_warning")}
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    print(json.dumps(study_status(args.study), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
