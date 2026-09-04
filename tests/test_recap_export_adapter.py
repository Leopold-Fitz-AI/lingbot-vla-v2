import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.recap_export_adapter import export_adapter


def _write_checkpoint(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.mkdir(parents=True)
    save_file(tensors, path / "model.safetensors")


def test_exports_only_adapter_and_verifies_frozen_base():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        base = {
            "model.layer.weight": torch.arange(6).reshape(2, 3),
            "model.layer.bias": torch.ones(2),
        }
        adapter = {
            "model.recap_velocity_lora_a": torch.ones(2, 1, 3),
            "model.recap_velocity_lora_b": torch.zeros(2, 2, 1),
        }
        _write_checkpoint(root / "base", base)
        _write_checkpoint(root / "tuned", {**base, **adapter})

        output = root / "export" / "recap_adapter.safetensors"
        summary = export_adapter(
            root / "tuned",
            output,
            base_checkpoint=root / "base",
            signed_velocity_axis=True,
        )

        assert summary["verification"] == {
            "verified_base_tensors": 2,
            "adapter_tensors": 2,
        }
        assert summary["signed_velocity_axis"] is True
        with safe_open(output, framework="pt", device="cpu") as handle:
            assert set(handle.keys()) == set(adapter)
            assert handle.metadata()["signed_velocity_axis"] == "true"
        sidecar = json.loads(
            output.with_suffix(output.suffix + ".json").read_text()
        )
        assert sidecar["artifact_sha256"] == summary["artifact_sha256"]


def test_rejects_changed_base_tensor():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        base = {"model.layer.weight": torch.zeros(2)}
        tuned = {
            "model.layer.weight": torch.ones(2),
            "model.recap_condition_embeddings": torch.zeros(2, 4),
        }
        _write_checkpoint(root / "base", base)
        _write_checkpoint(root / "tuned", tuned)

        with pytest.raises(ValueError, match="Frozen base verification failed"):
            export_adapter(
                root / "tuned",
                root / "adapter.safetensors",
                base_checkpoint=root / "base",
            )
