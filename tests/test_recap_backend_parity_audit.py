import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts import recap_audit_backend_parity as audit


def test_resource_gate_never_runs_nvidia_or_model_for_incomplete_study(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(json.dumps({"stage": "final_50_52", "state": "running"}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")

    def forbidden(*args, **kwargs):
        raise AssertionError("No command/model launch allowed while final study runs")

    monkeypatch.setattr(audit.subprocess, "check_output", forbidden)
    with pytest.raises(RuntimeError, match="not complete"):
        audit.require_idle_gpu(4, tmp_path)
    with pytest.raises(ValueError, match="authorized"):
        audit.require_idle_gpu(0, tmp_path)
    assert not (tmp_path / "started.json").exists()


def test_occupied_gpu_rejected_even_after_study_completion(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(json.dumps({"stage": "complete", "state": "not_proven"}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setattr(
        audit.subprocess,
        "check_output",
        lambda command, **kwargs: "512" if "--query-gpu=memory.used" in command else "23456",
    )
    with pytest.raises(RuntimeError, match="occupied"):
        audit.require_idle_gpu(4, tmp_path)


@pytest.mark.parametrize(
    "path,seed",
    [
        ("/study/evaluations/final/x.npz", 3500000),
        ("/data/x.npz", 5100000),
        ("/data/x.npz", 5200001),
        ("/data/x.npz", 5300001),
    ],
)
def test_final_inputs_forbidden(path, seed):
    with pytest.raises(ValueError, match="Final"):
        audit.ensure_consumed_input(path, seed)


def test_comparison_distinguishes_bitwise_identity_and_rejects_nan():
    a = np.zeros((1, 3), dtype=np.float32)
    b = a.copy()
    b[0, 1] = -0.0
    assert audit.compare_arrays(a, b)["rms"] == 0
    assert not audit.compare_arrays(a, b)["byte_identical"]
    b[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        audit.compare_arrays(a, b)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "TASK_STATES", {"hanging_mug": 2})
    root = tmp_path / "training"
    split = root / "state_splits/hanging_mug"
    split.mkdir(parents=True)
    (split / "split_summary.json").write_text(json.dumps({"train_states": [3500000], "holdout_states": [3600000]}))
    for partition, seed in (("train", 3500000), ("holdout", 3600000)):
        # Sorted first record is a failure. It MUST NOT be replaced by a success.
        for i in range(2):
            folder = split / partition / f"episode{i}"
            folder.mkdir(parents=True)
            np.savez(
                folder / "step.npz",
                generated_action=np.ones((50, 14), dtype=np.float32),
                executed_action=np.ones((50, 14), dtype=np.float32),
                **{"observation::observation.state": np.zeros(14)},
            )
            (folder / "manifest.json").write_text(
                json.dumps(
                    {
                        "task": "Hang the mug.",
                        "success": bool(i),
                        "steps": [{"file": "step.npz"}],
                        "metadata": {
                            "paired_environment_seed": seed,
                            "paired_match_id": f"{seed}:0",
                            "paired_original_decision_index": 1,
                            "continuation_policy_seed": 200,
                            "task_name": "hanging_mug",
                        },
                    }
                )
            )
    folder = root / "training_v2/hanging_mug/signed"
    (folder / "adapter").mkdir(parents=True)
    (folder / "train_config.yaml").write_text("train: {}\n")
    (folder / "adapter/recap_adapter.safetensors").write_bytes(b"only hashed during preparation")
    wrapper = tmp_path / "wrapper/checkpoints/global_step_50000/hf_ckpt"
    wrapper.mkdir(parents=True)
    (wrapper / "model.safetensors").write_bytes(b"only hashed during preparation")
    qwen = tmp_path / "qwen"
    qwen.mkdir()
    (qwen / "config.json").write_text("{}")
    norm = tmp_path / "norm.json"
    norm.write_text("{}")
    monkeypatch.setenv("QWEN3VL_PATH", str(qwen))
    (wrapper.parents[2] / "lingbotvla_cli.yaml").write_text(
        yaml.safe_dump({"data": {"norm_stats_file": str(norm)}, "model": {"tokenizer_path": str(qwen)}})
    )
    out = tmp_path / "probe"
    lock = audit.prepare(root, out, wrapper)
    return root, out, wrapper, lock


def test_prepare_freezes_full_grid_without_success_filtering(prepared):
    root, out, wrapper, lock = prepared
    assert lock["expected_points"] == 24
    assert all("episode0" in case["manifest"] for case in lock["cases"])
    assert audit.verify_lock(out) == lock
    with pytest.raises(FileExistsError, match="new diagnostic"):
        audit.prepare(root, out, wrapper)


def test_null_wrapper_normalizer_uses_actual_robot_config_fallback(prepared):
    root, out, wrapper, _ = prepared
    cli = wrapper.parents[2] / "lingbotvla_cli.yaml"
    config = yaml.safe_load(cli.read_text())
    config["data"]["norm_stats_file"] = None
    cli.write_text(yaml.safe_dump(config))
    lock = audit.prepare(root, out.parent / "fallback", wrapper)
    source = Path(audit.__file__).resolve().parents[1]
    expected = yaml.safe_load((source / "configs/robot_configs/robotwin.yaml").read_text())["norm_stats"]
    assert lock["norm"] == str((source / expected).resolve())


def test_modified_source_data_or_seal_is_rejected(prepared):
    _, out, _, lock = prepared
    data = Path(lock["cases"][0]["npz"])
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        audit.verify_lock(out)
    path = out / "input_lock.json"
    path.chmod(0o644)
    path.write_text("{}")
    with pytest.raises(ValueError, match="seal changed"):
        audit.verify_lock(out)


def test_incomplete_original_split_is_not_shortened(prepared):
    root, out, wrapper, lock = prepared
    path = Path(lock["cases"][0]["manifest"])
    path.unlink()
    (path.parent.parent / "episode1/manifest.json").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        audit.prepare(root, out.parent / "other", wrapper)


def test_invalid_recorded_chunks_fail_during_cpu_preparation(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, generated_action=np.zeros((50, 14)), executed_action=np.ones((50, 14)))
    with pytest.raises(ValueError, match="Invalid recorded"):
        audit.validate_raw_chunk(path)


def test_base_tensor_hash_checks_values_not_only_requires_grad():
    import torch

    model = torch.nn.Linear(3, 2)
    before = audit.base_tensor_hashes(model)
    model.requires_grad_(False)
    assert audit.base_tensor_hashes(model) == before
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert audit.base_tensor_hashes(model) != before
