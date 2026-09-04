#!/usr/bin/env python3
"""Convert raw RECAP decision bundles into a precomputed-chunk LeRobot dataset.

One LeRobot row represents one policy decision.  Its ``action`` value is the
complete fixed-length action chunk instead of a single action.  Train with
``data.precomputed_action_chunks: true`` so the VLA loader consumes this chunk
directly and does not query future rows across decision boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
VALID_LABELS = {-1, 0, 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--labels", required=True, help="Labeled episode JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id", default="local/recap-robotwin-decisions")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument(
        "--action-source",
        choices=("executed_action", "generated_action"),
        default="executed_action",
    )
    return parser.parse_args()


def load_labeled_decisions(path: str | Path) -> tuple[dict[tuple[str, int], dict[str, Any]], str]:
    label_path = Path(path).expanduser().resolve()
    labels: dict[tuple[str, int], dict[str, Any]] = {}
    with label_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            episode = json.loads(line)
            episode_id = str(episode.get("episode_id", ""))
            if not episode_id:
                raise ValueError(f"{label_path}:{line_number} is missing episode_id")
            steps = episode.get("steps")
            if not isinstance(steps, list) or not steps:
                raise ValueError(f"{label_path}:{line_number} has no labeled steps")
            for position, step in enumerate(steps):
                decision_index = int(step.get("decision_index", position))
                label = int(step["recap_label"])
                if label not in VALID_LABELS:
                    raise ValueError(f"Invalid recap_label {label} for {episode_id}/{decision_index}")
                key = (episode_id, decision_index)
                if key in labels:
                    raise ValueError(f"Duplicate labeled decision {key}")
                labels[key] = {
                    "recap_label": label,
                    "recap_advantage": float(step["recap_advantage"]),
                    "recap_return": float(step["recap_return"]),
                    "recap_value": float(step["value"]),
                }
    if not labels:
        raise ValueError(f"No labeled decisions found in {label_path}")
    return labels, hashlib.sha256(label_path.read_bytes()).hexdigest()


def pad_action_chunk(action: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 14:
        raise ValueError(f"Action chunk must have shape [T,14], got {action.shape}")
    if len(action) == 0 or len(action) > chunk_size:
        raise ValueError(f"Action chunk length must be in [1,{chunk_size}], got {len(action)}")
    if not np.isfinite(action).all():
        raise ValueError("Action chunk contains non-finite values")
    output = np.empty((chunk_size, action.shape[1]), dtype=np.float32)
    output[: len(action)] = action
    output[len(action) :] = action[-1]
    is_pad = np.zeros(chunk_size, dtype=np.bool_)
    is_pad[len(action) :] = True
    return output, is_pad


def prepare_rgb_image(
    image: np.ndarray,
    *,
    height: int = 240,
    width: int = 320,
) -> tuple[np.ndarray, bool]:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"RGB image must be uint8 HWC with 3 channels, got {image.shape}/{image.dtype}")
    if image.shape[:2] == (height, width):
        return image, False
    from PIL import Image

    resized = Image.fromarray(image).resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8).copy(), True


def _features(chunk_size: int) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
        "action": {"dtype": "float32", "shape": (chunk_size, 14), "names": None},
        "action_is_pad": {"dtype": "bool", "shape": (chunk_size,), "names": None},
        "recap_label": {"dtype": "int8", "shape": (1,), "names": ["label"]},
        "recap_advantage": {"dtype": "float32", "shape": (1,), "names": ["advantage"]},
        "recap_return": {"dtype": "float32", "shape": (1,), "names": ["return"]},
        "recap_value": {"dtype": "float32", "shape": (1,), "names": ["value"]},
    }
    for key in IMAGE_KEYS:
        features[key] = {
            "dtype": "image",
            "shape": (240, 320, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def convert(
    rollout_dir: str | Path,
    labels_path: str | Path,
    output: str | Path,
    *,
    repo_id: str,
    chunk_size: int,
    action_source: str,
) -> dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    rollout_root = Path(rollout_dir).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    manifests = sorted(rollout_root.rglob("manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No manifest.json files under {rollout_root}")
    labels, labels_sha256 = load_labeled_decisions(labels_path)

    temporary = output_root.with_name(f".{output_root.name}.recap-tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=1,
            features=_features(chunk_size),
            root=temporary,
            robot_type="aloha-agilex",
            use_videos=False,
            image_writer_processes=0,
            image_writer_threads=0,
        )
        matched: set[tuple[str, int]] = set()
        label_counts: Counter[int] = Counter()
        source_counts: Counter[str] = Counter()
        outcome_counts: Counter[str] = Counter()
        episode_rows: list[dict[str, Any]] = []
        decision_count = 0
        resized_image_count = 0

        for episode_index, manifest_path in enumerate(manifests):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            episode_id = str(manifest["episode_id"])
            task = str(manifest["task"])
            steps = manifest.get("steps")
            if not isinstance(steps, list) or not steps:
                raise ValueError(f"Episode {episode_id!r} has no decisions")
            if int(manifest.get("num_decisions", len(steps))) != len(steps):
                raise ValueError(f"Episode {episode_id!r} num_decisions does not match steps")

            for position, step in enumerate(steps):
                decision_index = int(step.get("decision_index", position))
                key = (episode_id, decision_index)
                if key not in labels:
                    raise ValueError(f"Missing label for decision {key}")
                annotation = labels[key]
                archive_path = manifest_path.parent / str(step["file"])
                with np.load(archive_path, allow_pickle=False) as archive:
                    action, action_is_pad = pad_action_chunk(archive[action_source], chunk_size)
                    state = np.asarray(
                        archive["observation::observation.state"], dtype=np.float32
                    ).reshape(-1)
                    if state.shape != (14,) or not np.isfinite(state).all():
                        raise ValueError(f"Invalid state in {archive_path}: {state.shape}")
                    images = {}
                    for image_key in IMAGE_KEYS:
                        image, was_resized = prepare_rgb_image(
                            archive[f"observation::{image_key}"]
                        )
                        images[image_key] = image
                        resized_image_count += int(was_resized)

                frame = {
                    "observation.state": state,
                    "action": action,
                    "action_is_pad": action_is_pad,
                    "recap_label": np.asarray([annotation["recap_label"]], dtype=np.int8),
                    "recap_advantage": np.asarray([annotation["recap_advantage"]], dtype=np.float32),
                    "recap_return": np.asarray([annotation["recap_return"]], dtype=np.float32),
                    "recap_value": np.asarray([annotation["recap_value"]], dtype=np.float32),
                    "task": task,
                    **images,
                }
                dataset.add_frame(frame)
                matched.add(key)
                label_counts[annotation["recap_label"]] += 1
                decision_count += 1

            dataset.save_episode()
            source = str(
                (manifest.get("metadata") or {}).get("source", "online_recap_policy")
            )
            success = bool(manifest.get("success", False))
            source_counts[source] += 1
            outcome_counts["success" if success else "failure"] += 1
            episode_rows.append(
                {
                    "episode_index": episode_index,
                    "source_episode_id": episode_id,
                    "source_manifest": str(manifest_path),
                    "task": task,
                    "success": success,
                    "source": source,
                    "decisions": len(steps),
                }
            )

        unmatched = sorted(set(labels) - matched)
        if unmatched:
            raise ValueError(f"{len(unmatched)} labels were not matched; first keys: {unmatched[:5]}")
        dataset.finalize()

        meta_dir = temporary / "meta"
        with (meta_dir / "recap_source_episodes.jsonl").open("w", encoding="utf-8") as output_file:
            for row in episode_rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "schema_version": 1,
            "repo_id": repo_id,
            "rollout_dir": str(rollout_root),
            "labels": str(Path(labels_path).expanduser().resolve()),
            "labels_sha256": labels_sha256,
            "action_source": action_source,
            "precomputed_action_chunks": True,
            "chunk_size": chunk_size,
            "episodes": len(episode_rows),
            "decisions": decision_count,
            "image_shape": [240, 320, 3],
            "resized_images": resized_image_count,
            "label_counts": {
                "null": label_counts[-1],
                "negative": label_counts[0],
                "positive": label_counts[1],
            },
            "sources": dict(source_counts),
            "outcomes": dict(outcome_counts),
        }
        (meta_dir / "recap_conversion.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output_root)
        summary["output"] = str(output_root)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> None:
    args = parse_args()
    result = convert(
        args.rollout_dir,
        args.labels,
        args.output,
        repo_id=args.repo_id,
        chunk_size=args.chunk_size,
        action_source=args.action_source,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
