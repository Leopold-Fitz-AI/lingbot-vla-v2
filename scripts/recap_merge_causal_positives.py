#!/usr/bin/env python3
"""Merge verified positive members of task-specific causal rollout pairs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def merge_causal_positives(
    inputs: list[Path],
    output: Path,
    *,
    task_name: str,
    minimum_positive: int = 1,
) -> dict:
    if minimum_positive <= 0:
        raise ValueError("minimum_positive must be positive")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    rows = []
    seen_originals = set()
    for root in inputs:
        root = root.expanduser().resolve()
        manifests = sorted(root.rglob("manifest.json"))
        if not manifests:
            raise FileNotFoundError(f"No manifests under {root}")
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metadata = manifest.get("metadata") or {}
            if metadata.get("task_name") != task_name or not manifest.get("success"):
                continue
            if metadata.get("source") != "paired_policy_sample":
                raise ValueError(f"Unpaired positive manifest: {manifest_path}")
            role = metadata.get("paired_match_role")
            if role is not None and role != "positive":
                raise ValueError(f"Positive outcome has non-positive pair role: {manifest_path}")
            original = str(metadata.get("paired_original_manifest", ""))
            if not original:
                raise ValueError(f"Missing paired_original_manifest: {manifest_path}")
            if original in seen_originals:
                raise ValueError(f"Duplicate paired source manifest: {original}")
            seen_originals.add(original)
            steps = manifest.get("steps")
            if not isinstance(steps, list) or len(steps) != 1:
                raise ValueError(f"Expected one paired decision: {manifest_path}")
            rows.append((root, manifest_path, manifest))
    if len(rows) < minimum_positive:
        raise ValueError(
            f"Only {len(rows)} causal positives for {task_name}; "
            f"minimum is {minimum_positive}"
        )

    temporary = output.with_name(f".{output.name}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    episode_rows = []
    try:
        for index, (root, manifest_path, manifest) in enumerate(rows):
            episode_dir = temporary / f"episode-{index:06d}"
            steps_dir = episode_dir / "steps"
            steps_dir.mkdir(parents=True)
            step = dict(manifest["steps"][0])
            source_npz = manifest_path.parent / str(step["file"])
            if not source_npz.is_file():
                raise FileNotFoundError(source_npz)
            step["file"] = "steps/000000.npz"
            step["decision_index"] = 0
            shutil.copy2(source_npz, episode_dir / step["file"])
            episode_id = f"causal-positive-{task_name}-{index:06d}"
            metadata = dict(manifest.get("metadata") or {})
            metadata.update(
                {
                    "causal_positive_merge_input": str(root),
                    "causal_positive_merge_manifest": str(manifest_path),
                }
            )
            merged = {
                **manifest,
                "episode_id": episode_id,
                "success": True,
                "num_decisions": 1,
                "metadata": metadata,
                "steps": [step],
            }
            (episode_dir / "manifest.json").write_text(
                json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "task": merged["task"],
                    "success": True,
                    "steps": [
                        {
                            "decision_index": 0,
                            "executed_action_length": step["executed_action_length"],
                            "terminated": step.get("terminated", False),
                            "valid": True,
                        }
                    ],
                }
            )
        with (temporary / "episodes.jsonl").open("w", encoding="utf-8") as handle:
            for episode in episode_rows:
                handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
        summary = {
            "schema_version": 1,
            "task_name": task_name,
            "inputs": [str(path.expanduser().resolve()) for path in inputs],
            "positive_episodes": len(rows),
            "output": str(output.resolve()),
        }
        (temporary / "merge_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--minimum-positive", type=int, default=1)
    args = parser.parse_args()
    result = merge_causal_positives(
        args.input,
        args.output,
        task_name=args.task,
        minimum_positive=args.minimum_positive,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
