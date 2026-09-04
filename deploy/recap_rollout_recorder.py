"""Dependency-light rollout recording for RECAP data collection.

This module intentionally depends only on NumPy and the standard library so it
can be copied into a RoboTwin simulation checkout alongside the WebSocket
client. Each policy decision is stored immediately as a compressed NPZ file;
an episode manifest is written atomically when the episode finishes.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    return name or "episode"


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy = getattr(value, "numpy", None)
        if callable(numpy):
            return numpy()
    return np.asarray(value)


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync for durable atomic renames."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some network filesystems do not implement directory fsync. The file
        # itself is still synced before rename.
        pass
    finally:
        os.close(descriptor)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    return str(value)


class RecapEpisodeRecorder:
    """Stream one simulation episode into an atomic on-disk bundle."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        episode_id: str,
        task: str,
        seed: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        episode_name = _safe_name(episode_id)
        self.episode_dir = root / episode_name
        if self.episode_dir.exists():
            raise FileExistsError(f"RECAP episode directory already exists: {self.episode_dir}")
        self.episode_dir.mkdir(parents=False)
        self.steps_dir = self.episode_dir / "steps"
        self.steps_dir.mkdir()
        self._finalized = False
        self._manifest: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "episode_id": str(episode_id),
            "task": str(task),
            "seed": None if seed is None else int(seed),
            "metadata": _json_safe(dict(metadata or {})),
            "steps": [],
        }

    @property
    def num_decisions(self) -> int:
        return len(self._manifest["steps"])

    def record_step(
        self,
        observation: Mapping[str, Any],
        generated_action: Any,
        executed_action: Any,
        *,
        env_step_before: int | None = None,
        env_step_after: int | None = None,
        terminated: bool = False,
        truncated: bool = False,
        intervention: bool = False,
        reward: float | None = None,
        info: Mapping[str, Any] | None = None,
    ) -> Path:
        if self._finalized:
            raise RuntimeError("Cannot append to a finalized RECAP episode")
        decision_index = self.num_decisions
        file_name = f"{decision_index:06d}.npz"
        path = self.steps_dir / file_name

        arrays: dict[str, np.ndarray] = {
            "generated_action": _to_numpy(generated_action),
            "executed_action": _to_numpy(executed_action),
        }
        observation_keys = []
        scalar_observation: dict[str, Any] = {}
        for key, value in observation.items():
            if isinstance(value, (str, bool, int, float, np.generic)):
                scalar_observation[str(key)] = _json_safe(value)
                continue
            array = _to_numpy(value)
            arrays[f"observation::{key}"] = array
            observation_keys.append(str(key))

        with tempfile.NamedTemporaryFile(
            dir=self.steps_dir,
            prefix=f".{file_name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with temporary_path.open("wb") as output:
                np.savez_compressed(output, **arrays)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, path)
            _fsync_directory(self.steps_dir)
        finally:
            temporary_path.unlink(missing_ok=True)

        generated = arrays["generated_action"]
        executed = arrays["executed_action"]
        generated_length = int(generated.shape[0]) if generated.ndim >= 2 else 1
        executed_length = int(executed.shape[0]) if executed.ndim >= 2 else 1
        step_manifest = {
            "decision_index": decision_index,
            "file": str(Path("steps") / file_name),
            "observation_array_keys": observation_keys,
            "observation_scalars": scalar_observation,
            "generated_action_length": generated_length,
            "executed_action_length": executed_length,
            "env_step_before": None if env_step_before is None else int(env_step_before),
            "env_step_after": None if env_step_after is None else int(env_step_after),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "intervention": bool(intervention),
            "reward": None if reward is None else float(reward),
            "info": _json_safe(dict(info or {})),
        }
        self._manifest["steps"].append(step_manifest)
        return path

    def finalize(
        self,
        *,
        success: bool,
        terminal_reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        if self._finalized:
            raise RuntimeError("RECAP episode was already finalized")
        self._manifest["success"] = bool(success)
        self._manifest["terminal_reason"] = str(terminal_reason)
        self._manifest["num_decisions"] = self.num_decisions
        if metadata:
            self._manifest["metadata"].update(_json_safe(dict(metadata)))

        path = self.episode_dir / "manifest.json"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.episode_dir,
            prefix=".manifest.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(self._manifest, temporary, ensure_ascii=False, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, path)
            _fsync_directory(self.episode_dir)
            # Detect silent zero-filled writes observed on some shared NFS
            # mounts before treating the episode as complete.
            persisted = json.loads(path.read_text(encoding="utf-8"))
            if persisted.get("episode_id") != self._manifest["episode_id"]:
                raise OSError(f"Manifest verification failed after atomic rename: {path}")
        finally:
            temporary_path.unlink(missing_ok=True)
        self._finalized = True
        return path
