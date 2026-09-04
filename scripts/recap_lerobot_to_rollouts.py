#!/usr/bin/env python3
"""Convert successful LeRobot demonstrations into raw RECAP rollout bundles.

Each policy decision begins at one demonstration frame and contains at most
``chunk_size`` contiguous expert actions.  Generated and executed actions are
identical because these are accepted expert trajectories.  The resulting
bundles use the same schema as ``deploy/recap_rollout_recorder.py`` and can be
combined with failed online-policy rollouts for value fitting.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.recap_rollout_recorder import RecapEpisodeRecorder  # noqa: E402


IMAGE_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="LeRobot dataset root")
    parser.add_argument("--output", required=True, help="Raw RECAP rollout output root")
    parser.add_argument("--task-name", default="adjust_bottle")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--run-id", default="expert-demo")
    return parser.parse_args()


def _decode_image(value: dict[str, Any], dataset_root: Path) -> np.ndarray:
    from PIL import Image

    payload = value.get("bytes")
    if payload is not None:
        source: Any = io.BytesIO(payload)
    else:
        relative_path = value.get("path")
        if not relative_path:
            raise ValueError("Image struct contains neither bytes nor path")
        source = dataset_root / relative_path
    with Image.open(source) as image:
        return np.asarray(image.convert("RGB")).copy()


def _load_tables(dataset_root: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_paths = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No data parquet files found under {dataset_root / 'data'}")
    columns = ["observation.state", "action", *IMAGE_KEYS, "frame_index", "episode_index", "index", "task_index"]
    tables = [pq.read_table(path, columns=columns) for path in data_paths]
    table = pa.concat_tables(tables)
    order = np.argsort(np.asarray(table["index"].to_numpy()), kind="stable")
    table = table.take(pa.array(order))

    tasks_path = dataset_root / "meta" / "tasks.parquet"
    tasks = pq.read_table(tasks_path).to_pydict()
    task_text_key = "task" if "task" in tasks else "__index_level_0__"
    task_map = {
        int(task_index): str(task)
        for task_index, task in zip(tasks["task_index"], tasks[task_text_key])
    }
    return table, task_map


def convert_dataset(
    dataset_root: Path,
    output_root: Path,
    *,
    task_name: str,
    chunk_size: int,
    max_episodes: int | None,
    run_id: str,
) -> dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    table, task_map = _load_tables(dataset_root)
    episode_indices = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    unique_episodes = list(dict.fromkeys(episode_indices.tolist()))
    if max_episodes is not None:
        unique_episodes = unique_episodes[:max_episodes]

    episodes_written = 0
    decisions_written = 0
    frames_written = 0
    for episode_index in unique_episodes:
        row_indices = np.flatnonzero(episode_indices == episode_index)
        episode = table.take(row_indices)
        frame_indices = np.asarray(episode["frame_index"].to_numpy(), dtype=np.int64)
        if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
            raise ValueError(f"Episode {episode_index} frame indices are not contiguous from zero")

        task_indices = set(np.asarray(episode["task_index"].to_numpy(), dtype=np.int64).tolist())
        if len(task_indices) != 1:
            raise ValueError(f"Episode {episode_index} contains multiple task indices: {task_indices}")
        task_index = next(iter(task_indices))
        task = task_map[task_index]
        episode_id = f"{task_name}-{run_id}-episode-{episode_index:06d}"
        recorder = RecapEpisodeRecorder(
            output_root / task_name,
            episode_id=episode_id,
            task=task,
            seed=None,
            metadata={
                "task_name": task_name,
                "source": "successful_lerobot_expert_demonstration",
                "source_dataset": str(dataset_root),
                "source_episode_index": int(episode_index),
                "recap_condition": "positive",
                "action_type": "joint",
                "state_dim": 14,
                "action_dim": 14,
                "chunk_size": int(chunk_size),
            },
        )

        actions = np.asarray(episode["action"].to_pylist(), dtype=np.float32)
        states = np.asarray(episode["observation.state"].to_pylist(), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14 or states.shape != actions.shape:
            raise ValueError(
                f"Episode {episode_index} requires matching [T,14] state/action arrays; "
                f"got state={states.shape}, action={actions.shape}"
            )
        if not np.isfinite(actions).all() or not np.isfinite(states).all():
            raise ValueError(f"Episode {episode_index} contains non-finite state/action values")

        for start in range(0, len(actions), chunk_size):
            stop = min(start + chunk_size, len(actions))
            observation = {"observation.state": states[start]}
            for image_key in IMAGE_KEYS:
                observation[image_key] = _decode_image(episode[image_key][start].as_py(), dataset_root)
            action_chunk = actions[start:stop]
            is_final = stop == len(actions)
            recorder.record_step(
                observation,
                action_chunk,
                action_chunk,
                env_step_before=start,
                env_step_after=stop,
                terminated=is_final,
                truncated=False,
                info={"source_frame_index": int(start), "source_task_index": int(task_index)},
            )
            decisions_written += 1

        recorder.finalize(
            success=True,
            terminal_reason="successful_expert_demonstration",
            metadata={"environment_steps": len(actions)},
        )
        episodes_written += 1
        frames_written += len(actions)

    return {
        "episodes": episodes_written,
        "decisions": decisions_written,
        "frames": frames_written,
        "output": str(output_root),
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary = convert_dataset(
        dataset_root,
        output_root,
        task_name=args.task_name,
        chunk_size=args.chunk_size,
        max_episodes=args.max_episodes,
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
