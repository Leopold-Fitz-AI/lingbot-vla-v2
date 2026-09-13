#!/usr/bin/env python3
"""Merge RECAP rollout runs into one directory with unique episode ids.

Policy-variant studies record the same logical episode under several
condition branches (``positive/``, ``null/``, ...), reusing the episode id.
``load_rollout_decisions`` requires globally unique episode ids, so this
tool materializes a merged directory in which every manifest is rewritten
with an id prefixed by its source root and branch directory. Step NPZs are
symlinked, not copied.

Example:

    python scripts/recap_merge_rollout_runs.py \
      --input /data/run_a/rollouts --input /data/run_b/rollouts \
      --output /data/merged_rollouts
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _safe_component(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    return name or "run"


def merge_rollout_runs(
    inputs: list[str | Path],
    output: str | Path,
) -> dict[str, int]:
    """Rewrite manifests under ``output`` with unique, source-prefixed ids."""

    roots = [Path(item).expanduser().resolve() for item in inputs]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Rollout input does not exist: {root}")
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Output directory already exists: {output_path}")

    planned: list[tuple[Path, dict, str, str]] = []
    seen_ids: set[str] = set()
    for root in roots:
        source_tag = _safe_component(root.name)
        for manifest_path in sorted(root.rglob("manifest.json")):
            with manifest_path.open(encoding="utf-8") as file:
                manifest = json.load(file)
            old_id = str(manifest.get("episode_id", ""))
            if not old_id:
                raise ValueError(f"Missing episode_id in {manifest_path}")
            relative = manifest_path.parent.relative_to(root)
            branch = _safe_component(relative.parts[0]) if relative.parts else "root"
            new_id = f"{source_tag}-{branch}-{_safe_component(old_id)}"
            suffix = 1
            candidate = new_id
            while candidate in seen_ids:
                suffix += 1
                candidate = f"{new_id}-dup{suffix}"
            seen_ids.add(candidate)
            planned.append((manifest_path, manifest, candidate, source_tag))

    if not planned:
        raise ValueError("No manifests found in the provided inputs")

    episodes = 0
    decisions = 0
    output_path.mkdir(parents=True)
    try:
        for manifest_path, manifest, new_id, _source_tag in planned:
            episode_dir = output_path / new_id
            episode_dir.mkdir()
            steps = manifest.get("steps")
            if not isinstance(steps, list) or not steps:
                raise ValueError(f"Episode {manifest.get('episode_id')!r} has no steps")
            steps_target = episode_dir / "steps"
            steps_target.mkdir()
            for step in steps:
                source_file = manifest_path.parent / str(step["file"])
                if not source_file.is_file():
                    raise FileNotFoundError(f"Missing step file: {source_file}")
                os.symlink(source_file, steps_target / Path(str(step["file"])).name)
                decisions += 1
            manifest = dict(manifest)
            manifest["episode_id"] = new_id
            metadata = dict(manifest.get("metadata") or {})
            metadata["merged_from"] = str(manifest_path.parent)
            manifest["metadata"] = metadata
            with (episode_dir / "manifest.json").open("w", encoding="utf-8") as file:
                json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
                file.write("\n")
            episodes += 1
    except Exception:
        # Fail closed: a partial merge must not be mistaken for a dataset.
        import shutil

        shutil.rmtree(output_path, ignore_errors=True)
        raise
    return {"episodes": episodes, "decisions": decisions, "output": str(output_path)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Rollout root; repeatable")
    parser.add_argument("--output", required=True, help="Merged output directory (must not exist)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = merge_rollout_runs(args.input, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
