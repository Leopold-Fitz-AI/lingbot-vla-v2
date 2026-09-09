#!/usr/bin/env python3
"""Export and verify a compact RECAP adapter from a full HF checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_map(checkpoint: Path) -> dict[str, Path]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as handle:
            index = json.load(handle)
        return {
            name: checkpoint / filename
            for name, filename in index["weight_map"].items()
        }

    result: dict[str, Path] = {}
    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors found under {checkpoint}")
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in result:
                    raise ValueError(f"Duplicate tensor {name!r} in {checkpoint}")
                result[name] = path
    return result


def _is_recap_tensor(name: str) -> bool:
    return any(part.startswith("recap_") for part in name.split("."))


def _open_handles(paths: set[Path], stack: ExitStack):
    return {
        path: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
        for path in paths
    }


def verify_frozen_base(
    checkpoint_map: dict[str, Path], base_map: dict[str, Path]
) -> dict[str, int]:
    adapter_names = {name for name in checkpoint_map if _is_recap_tensor(name)}
    checkpoint_base_names = set(checkpoint_map) - adapter_names
    if checkpoint_base_names != set(base_map):
        missing = sorted(set(base_map) - checkpoint_base_names)
        unexpected = sorted(checkpoint_base_names - set(base_map))
        raise ValueError(
            "Base tensor names differ: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    mismatches: list[str] = []
    with ExitStack() as stack:
        current_handles = _open_handles(set(checkpoint_map.values()), stack)
        base_handles = _open_handles(set(base_map.values()), stack)
        for name in sorted(base_map):
            current = current_handles[checkpoint_map[name]].get_tensor(name)
            expected = base_handles[base_map[name]].get_tensor(name)
            if not torch.equal(current, expected):
                mismatches.append(name)
                if len(mismatches) == 10:
                    break
    if mismatches:
        raise ValueError(
            "Frozen base verification failed for tensor(s): " + ", ".join(mismatches)
        )
    return {
        "verified_base_tensors": len(base_map),
        "adapter_tensors": len(adapter_names),
    }


def export_adapter(
    checkpoint: Path,
    output: Path,
    *,
    base_checkpoint: Path | None = None,
    signed_velocity_axis: bool = False,
) -> dict:
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    checkpoint_map = _tensor_map(checkpoint)
    adapter_names = sorted(name for name in checkpoint_map if _is_recap_tensor(name))
    if not adapter_names:
        raise ValueError(f"No RECAP tensors found in {checkpoint}")

    verification = None
    if base_checkpoint is not None:
        base_checkpoint = base_checkpoint.resolve()
        verification = verify_frozen_base(
            checkpoint_map,
            _tensor_map(base_checkpoint),
        )

    with ExitStack() as stack:
        handles = _open_handles({checkpoint_map[name] for name in adapter_names}, stack)
        tensors = {
            name: handles[checkpoint_map[name]].get_tensor(name).contiguous()
            for name in adapter_names
        }

    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite adapter tensor: {name}")
    training_configuration = None
    config_path = checkpoint / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        if "recap_training_backend" in config or "recap_adapter_initialization" in config:
            training_configuration = {
                "config_sha256": _sha256(config_path),
                "configured_initialization": config.get("recap_adapter_initialization", "legacy_sin_v1"),
                "initialization_seed": config.get("recap_adapter_init_seed", 0),
                "initialization_scale": config.get("recap_adapter_init_std", 0.02),
                "training_backend": config.get("recap_training_backend", "legacy"),
                "tensor_dtypes": sorted({str(t.dtype) for t in tensors.values()}),
                # Config provenance is not proof that a GPU parity gate passed,
                # or that loaded/resumed weights were reset using this scheme.
                "scope": "checkpoint configuration, not a runtime parity attestation",
            }

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        save_file(
            tensors,
            temporary,
            metadata={
                "format": "pt",
                "schema": "lingbotvla-recap-adapter-v1",
                "signed_velocity_axis": str(bool(signed_velocity_axis)).lower(),
                **({"recap_training_configuration": json.dumps(training_configuration, sort_keys=True)}
                   if training_configuration is not None else {}),
            },
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)

    summary = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "base_checkpoint": None if base_checkpoint is None else str(base_checkpoint),
        "artifact": str(output),
        "artifact_sha256": _sha256(output),
        "signed_velocity_axis": bool(signed_velocity_axis),
        "tensors": {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
            }
            for name, tensor in tensors.items()
        },
        "verification": verification,
        "training_configuration": training_configuration,
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_bytes = (
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    ).encode()
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{metadata_path.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(metadata_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, metadata_path)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signed-velocity-axis", action="store_true")
    args = parser.parse_args()
    summary = export_adapter(
        args.checkpoint,
        args.output,
        base_checkpoint=args.base_checkpoint,
        signed_velocity_axis=args.signed_velocity_axis,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
