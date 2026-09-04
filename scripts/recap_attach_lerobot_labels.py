#!/usr/bin/env python3
"""Attach RECAP labels to a cloned LeRobot v2.1/v3 dataset.

The labeled JSONL must identify each training frame with ``episode_index`` and
``frame_index``. For decision-level datasets whose frame index is exactly the
policy decision index, pass ``--use-decision-index`` explicitly.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


VALID_LABELS = {-1, 0, 1}


def load_label_map(
    path: str | Path,
    *,
    use_decision_index: bool = False,
) -> dict[tuple[int, int], int]:
    labels: dict[tuple[int, int], int] = {}
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            episode = json.loads(line)
            episode_index = episode.get("episode_index")
            if episode_index is None:
                episode_id = episode.get("episode_id")
                if isinstance(episode_id, int) or str(episode_id).isdigit():
                    episode_index = int(episode_id)
                else:
                    raise ValueError(f"{path}:{line_number} must provide numeric episode_index")
            steps = episode.get("steps")
            if not isinstance(steps, list):
                raise ValueError(f"{path}:{line_number} has no steps list")
            for step in steps:
                frame_index = step.get("frame_index")
                if frame_index is None and use_decision_index:
                    frame_index = step.get("decision_index")
                if frame_index is None:
                    raise ValueError(
                        f"Episode {episode_index} step must provide frame_index; "
                        "use --use-decision-index only when both indices are identical"
                    )
                label = int(step["recap_label"])
                if label not in VALID_LABELS:
                    raise ValueError(f"Invalid recap_label {label}; expected -1, 0, or 1")
                key = (int(episode_index), int(frame_index))
                if key in labels:
                    raise ValueError(f"Duplicate RECAP label for episode/frame {key}")
                labels[key] = label
    if not labels:
        raise ValueError(f"No RECAP labels found in {path}")
    return labels


def _missing_default(mode: str) -> int | None:
    return {
        "positive": 1,
        "negative": 0,
        "null": -1,
        "error": None,
        "preserve": None,
    }[mode]


def _update_info_json(dataset_root: Path, column: str) -> None:
    path = dataset_root / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {path}")
    with path.open(encoding="utf-8") as source:
        info = json.load(source)
    features = info.setdefault("features", {})
    features[column] = {"dtype": "int8", "shape": [1], "names": None}
    temporary_path = path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(info, output, ensure_ascii=False, indent=2)
        output.write("\n")
    temporary_path.replace(path)


def attach_labels(
    source_root: str | Path,
    output_root: str | Path,
    labels: dict[tuple[int, int], int],
    *,
    column: str = "recap_label",
    missing: str = "error",
    allow_unmatched_labels: bool = False,
) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "recap_attach_lerobot_labels.py requires pyarrow; install the project's "
            "LeRobot/datasets training dependencies"
        ) from exc

    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"LeRobot dataset directory does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Output dataset already exists: {output}")
    if source == output or source in output.parents:
        raise ValueError("Output dataset must not be inside the source dataset")
    temporary_output = output.with_name(f".{output.name}.recap-tmp")
    if temporary_output.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_output}")
    shutil.copytree(source, temporary_output)

    parquet_files = sorted((temporary_output / "data").rglob("*.parquet"))
    if not parquet_files:
        shutil.rmtree(temporary_output)
        raise FileNotFoundError(f"No LeRobot parquet files found under {temporary_output / 'data'}")
    default_label = _missing_default(missing)
    matched: set[tuple[int, int]] = set()
    rows_written = 0
    label_counts = {-1: 0, 0: 0, 1: 0}

    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        required = {"episode_index", "frame_index"}
        if not required.issubset(table.column_names):
            raise ValueError(f"{parquet_path} is missing episode_index/frame_index")
        episode_indices = table["episode_index"].to_pylist()
        frame_indices = table["frame_index"].to_pylist()
        existing = table[column].to_pylist() if column in table.column_names else None
        column_values = []
        for row_index, (episode_index, frame_index) in enumerate(zip(episode_indices, frame_indices)):
            key = (int(episode_index), int(frame_index))
            if key in labels:
                label = labels[key]
                matched.add(key)
            elif missing == "preserve" and existing is not None:
                label = int(existing[row_index])
            elif default_label is not None:
                label = default_label
            else:
                raise ValueError(f"No RECAP label for episode/frame {key} in {parquet_path}")
            if label not in VALID_LABELS:
                raise ValueError(f"Existing {column} value {label} is invalid")
            column_values.append(label)
            label_counts[label] += 1
        label_array = pa.array(column_values, type=pa.int8())
        if column in table.column_names:
            table = table.set_column(table.schema.get_field_index(column), column, label_array)
        else:
            table = table.append_column(column, label_array)
        temporary_path = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary_path)
        temporary_path.replace(parquet_path)
        rows_written += table.num_rows

    unmatched = sorted(set(labels) - matched)
    if unmatched and not allow_unmatched_labels:
        shutil.rmtree(temporary_output)
        raise ValueError(f"{len(unmatched)} labels did not match a LeRobot row; first keys: {unmatched[:5]}")
    _update_info_json(temporary_output, column)
    temporary_output.replace(output)
    return {
        "output": str(output),
        "parquet_files": len(parquet_files),
        "rows": rows_written,
        "matched_labels": len(matched),
        "unmatched_labels": len(unmatched),
        "label_counts": {
            "null": label_counts[-1],
            "negative": label_counts[0],
            "positive": label_counts[1],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--labels", required=True, help="Labeled episode JSONL")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--column", default="recap_label")
    parser.add_argument(
        "--missing",
        choices=("error", "positive", "negative", "null", "preserve"),
        default="error",
        help="How to label dataset rows absent from the JSONL map",
    )
    parser.add_argument("--use-decision-index", action="store_true")
    parser.add_argument("--allow-unmatched-labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_label_map(
        args.labels,
        use_decision_index=args.use_decision_index,
    )
    result = attach_labels(
        args.dataset_root,
        args.output_root,
        labels,
        column=args.column,
        missing=args.missing,
        allow_unmatched_labels=args.allow_unmatched_labels,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
