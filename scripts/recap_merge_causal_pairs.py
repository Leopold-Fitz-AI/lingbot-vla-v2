#!/usr/bin/env python3
"""Merge state-balanced success/failure pairs for signed RECAP training."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def merge_causal_pairs(
    inputs: list[Path],
    output: Path,
    *,
    task_name: str,
    maximum_pairs_per_state: int = 2,
    minimum_states: int = 1,
    minimum_pairs: int = 1,
) -> dict:
    if maximum_pairs_per_state <= 0 or minimum_states <= 0 or minimum_pairs <= 0:
        raise ValueError("pair/state gates must be positive")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    grouped: dict[tuple[str, int, str], list[tuple[Path, dict]]] = defaultdict(list)
    for root_value in inputs:
        root = root_value.expanduser().resolve()
        manifests = sorted(root.rglob("manifest.json"))
        if not manifests:
            raise FileNotFoundError(f"No manifests under {root}")
        for path in manifests:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            metadata = manifest.get("metadata") or {}
            if metadata.get("task_name") != task_name:
                continue
            if metadata.get("source") != "paired_policy_sample":
                raise ValueError(f"Unpaired causal manifest: {path}")
            pair_id = metadata.get("paired_match_id")
            seed = metadata.get("paired_environment_seed")
            if pair_id is None or seed is None:
                raise ValueError(f"Missing pair identity: {path}")
            if len(manifest.get("steps") or []) != 1:
                raise ValueError(f"Expected one causal decision: {path}")
            grouped[(str(root), int(seed), str(pair_id))].append((path, manifest))

    by_state = defaultdict(list)
    seen_originals = set()
    for (root, seed, pair_id), members in grouped.items():
        if len(members) != 2:
            raise ValueError(
                f"Causal pair {root}/{pair_id} has {len(members)} members, expected 2"
            )
        outcomes = {bool(manifest.get("success")) for _, manifest in members}
        roles = {
            (manifest.get("metadata") or {}).get("paired_match_role")
            for _, manifest in members
        }
        if outcomes != {False, True} or roles != {"negative", "positive"}:
            raise ValueError(f"Invalid signed causal pair {root}/{pair_id}")
        originals = [
            str((manifest.get("metadata") or {}).get("paired_original_manifest", ""))
            for _, manifest in members
        ]
        if not all(originals) or any(value in seen_originals for value in originals):
            raise ValueError(f"Missing or duplicate original manifest in {root}/{pair_id}")
        seen_originals.update(originals)
        drift = max(
            float((manifest.get("metadata") or {}).get("paired_numeric_max_abs", 0.0))
            + float((manifest.get("metadata") or {}).get("paired_image_mae_max", 0.0))
            for _, manifest in members
        )
        by_state[seed].append((drift, root, pair_id, members))

    selected = []
    for seed, pairs in sorted(by_state.items()):
        selected.extend(
            (seed, *pair)
            for pair in sorted(pairs, key=lambda value: value[:3])[
                :maximum_pairs_per_state
            ]
        )
    selected_states = len({row[0] for row in selected})
    if selected_states < minimum_states:
        raise ValueError(
            f"Only {selected_states} independent causal states; minimum is {minimum_states}"
        )
    if len(selected) < minimum_pairs:
        raise ValueError(f"Only {len(selected)} causal pairs; minimum is {minimum_pairs}")

    temporary = output.with_name(f".{output.name}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    episodes = []
    try:
        output_index = 0
        for seed, drift, root, pair_id, members in selected:
            for manifest_path, manifest in sorted(
                members, key=lambda value: not bool(value[1].get("success"))
            ):
                episode_dir = temporary / f"episode-{output_index:06d}"
                (episode_dir / "steps").mkdir(parents=True)
                step = dict(manifest["steps"][0])
                source_npz = manifest_path.parent / str(step["file"])
                if not source_npz.is_file():
                    raise FileNotFoundError(source_npz)
                step["decision_index"] = 0
                step["file"] = "steps/000000.npz"
                shutil.copy2(source_npz, episode_dir / step["file"])
                success = bool(manifest["success"])
                episode_id = (
                    f"causal-pair-{task_name}-{seed}-{output_index:06d}-"
                    f"{'positive' if success else 'negative'}"
                )
                metadata = dict(manifest.get("metadata") or {})
                metadata.update(
                    {
                        "causal_pair_merge_input": root,
                        "causal_pair_merge_manifest": str(manifest_path),
                    }
                )
                merged = {
                    **manifest,
                    "episode_id": episode_id,
                    "num_decisions": 1,
                    "metadata": metadata,
                    "steps": [step],
                }
                (episode_dir / "manifest.json").write_text(
                    json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "task": merged["task"],
                        "success": success,
                        "steps": [
                            {
                                "decision_index": 0,
                                "executed_action_length": step[
                                    "executed_action_length"
                                ],
                                "terminated": step.get("terminated", False),
                                "valid": True,
                            }
                        ],
                    }
                )
                output_index += 1
        with (temporary / "episodes.jsonl").open("w", encoding="utf-8") as handle:
            for episode in episodes:
                handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
        summary = {
            "schema_version": 1,
            "task_name": task_name,
            "inputs": [str(path.expanduser().resolve()) for path in inputs],
            "independent_states": selected_states,
            "pairs": len(selected),
            "positive": len(selected),
            "negative": len(selected),
            "maximum_pairs_per_state": maximum_pairs_per_state,
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
    parser.add_argument("--maximum-pairs-per-state", type=int, default=2)
    parser.add_argument("--minimum-states", type=int, default=1)
    parser.add_argument("--minimum-pairs", type=int, default=1)
    args = parser.parse_args()
    summary = merge_causal_pairs(
        args.input,
        args.output,
        task_name=args.task,
        maximum_pairs_per_state=args.maximum_pairs_per_state,
        minimum_states=args.minimum_states,
        minimum_pairs=args.minimum_pairs,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
