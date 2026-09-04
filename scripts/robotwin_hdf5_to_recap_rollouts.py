#!/usr/bin/env python3
"""Convert successful RoboTwin HDF5 demonstrations to RECAP decision bundles.

The official RoboTwin files store observations and next-state joint actions at
50 Hz. This tool samples one observation per policy chunk and stores the next
``chunk_size`` joint targets in the same bundle format as online RECAP
rollouts. Every converted episode is marked successful, but labels are created
separately so the conversion remains auditable.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


CAMERA_MAP = {
    "head_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}


def _episode_number(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)\.hdf5", path.name)
    if match is None:
        raise ValueError(f"Unexpected RoboTwin episode filename: {path.name}")
    return int(match.group(1))


def _decode_image(value: np.ndarray, *, width: int, height: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(value.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode a RoboTwin camera frame")
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    # RoboTwin's official conversion preserves OpenCV's BGR byte order. Keep
    # the same convention here so expert replay matches the official corpus.
    return np.asarray(image, dtype=np.uint8)


def _states(handle: h5py.File) -> np.ndarray:
    left_arm = np.asarray(handle["/joint_action/left_arm"], dtype=np.float32)
    left_gripper = np.asarray(handle["/joint_action/left_gripper"], dtype=np.float32)[:, None]
    right_arm = np.asarray(handle["/joint_action/right_arm"], dtype=np.float32)
    right_gripper = np.asarray(handle["/joint_action/right_gripper"], dtype=np.float32)[:, None]
    states = np.concatenate((left_arm, left_gripper, right_arm, right_gripper), axis=1)
    if states.ndim != 2 or states.shape[1] != 14 or not np.isfinite(states).all():
        raise ValueError(f"Invalid joint trajectory shape/content: {states.shape}")
    return states


def _instruction(raw_root: Path, episode_number: int, split: str, index: int) -> str:
    path = raw_root / "instructions" / f"episode{episode_number}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    choices = data.get(split)
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"{path} has no non-empty {split!r} instruction list")
    return str(choices[index % len(choices)])


def convert_episode(
    source: Path,
    output: Path,
    *,
    task: str,
    chunk_size: int,
    image_width: int,
    image_height: int,
    instruction: str,
) -> dict[str, Any]:
    episode_number = _episode_number(source)
    episode_id = f"{task}-expert-episode-{episode_number}"
    episode_dir = output / f"episode-{episode_number:06d}"
    steps_dir = episode_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=False)

    manifest_steps: list[dict[str, Any]] = []
    episode_steps: list[dict[str, Any]] = []
    with h5py.File(source, "r") as handle:
        states = _states(handle)
        if len(states) < 2:
            raise ValueError(f"{source} has fewer than two states")
        cameras = {
            source_name: handle[f"/observation/{source_name}/rgb"]
            for source_name in CAMERA_MAP
        }
        if any(len(frames) != len(states) for frames in cameras.values()):
            raise ValueError(f"Camera/state length mismatch in {source}")

        decision_index = 0
        for state_index in range(0, len(states) - 1, chunk_size):
            action = states[state_index + 1 : state_index + 1 + chunk_size]
            executed_length = len(action)
            padded = np.empty((chunk_size, 14), dtype=np.float32)
            padded[:executed_length] = action
            padded[executed_length:] = action[-1]
            arrays: dict[str, np.ndarray] = {
                # ``generated_action`` is the fixed policy-sized proposal;
                # ``executed_action`` must contain only real actions. Keeping
                # the latter unpadded lets the LeRobot converter create an
                # accurate action_is_pad mask for the terminal chunk.
                "generated_action": padded,
                "executed_action": np.asarray(action, dtype=np.float32),
                "observation::observation.state": states[state_index],
            }
            for source_name, feature_name in CAMERA_MAP.items():
                arrays[f"observation::{feature_name}"] = _decode_image(
                    cameras[source_name][state_index],
                    width=image_width,
                    height=image_height,
                )

            relative = f"steps/{decision_index:06d}.npz"
            np.savez_compressed(episode_dir / relative, **arrays)
            is_last = state_index + chunk_size >= len(states) - 1
            step = {
                "decision_index": decision_index,
                "env_step_before": state_index,
                "env_step_after": state_index + executed_length,
                "executed_action_length": executed_length,
                "generated_action_length": chunk_size,
                "file": relative,
                "intervention": False,
                "observation_array_keys": list(CAMERA_MAP.values()) + ["observation.state"],
                "observation_scalars": {"task": instruction},
                "reward": None,
                "terminated": is_last,
                "truncated": False,
            }
            manifest_steps.append(step)
            episode_steps.append(
                {
                    "decision_index": decision_index,
                    "executed_action_length": executed_length,
                    "terminated": is_last,
                    "valid": True,
                }
            )
            decision_index += 1

    manifest = {
        "schema_version": 1,
        "episode_id": episode_id,
        "task": instruction,
        "success": True,
        "terminal_reason": "expert_demonstration",
        "seed": None,
        "num_decisions": len(manifest_steps),
        "metadata": {
            "source": "official_robotwin_expert",
            "source_hdf5": str(source.resolve()),
            "task_name": task,
            "task_config": "aloha-agilex_clean_50",
            "environment_steps": len(states) - 1,
        },
        "steps": manifest_steps,
    }
    (episode_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "episode_id": episode_id,
        "task": instruction,
        "success": True,
        "steps": episode_steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Extracted aloha-agilex_clean_50 directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--instruction-split", default="seen")
    parser.add_argument("--instruction-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.chunk_size <= 0:
        raise ValueError("episodes and chunk-size must be positive")
    raw_root = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    files = sorted((raw_root / "data").glob("episode*.hdf5"), key=_episode_number)
    if len(files) < args.episodes:
        raise ValueError(f"Requested {args.episodes} episodes, found {len(files)}")

    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    episodes: list[dict[str, Any]] = []
    try:
        for source in files[: args.episodes]:
            number = _episode_number(source)
            episodes.append(
                convert_episode(
                    source,
                    temporary,
                    task=args.task,
                    chunk_size=args.chunk_size,
                    image_width=args.image_width,
                    image_height=args.image_height,
                    instruction=_instruction(
                        raw_root, number, args.instruction_split, args.instruction_index
                    ),
                )
            )
        with (temporary / "episodes.jsonl").open("w", encoding="utf-8") as file:
            for episode in episodes:
                file.write(json.dumps(episode, ensure_ascii=False) + "\n")
        summary = {
            "schema_version": 1,
            "source": str(raw_root),
            "task": args.task,
            "episodes": len(episodes),
            "decisions": sum(len(episode["steps"]) for episode in episodes),
            "chunk_size": args.chunk_size,
            "image_shape": [args.image_height, args.image_width, 3],
            "instruction_split": args.instruction_split,
            "instruction_index": args.instruction_index,
        }
        (temporary / "conversion.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({**summary, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
