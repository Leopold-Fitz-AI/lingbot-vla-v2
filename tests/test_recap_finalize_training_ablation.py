import pytest
import torch
from safetensors.torch import save_file

from lingbotvla.recap.ablation import BACKENDS, INITIALIZATIONS, SEEDS, TASKS, sample_schedule, summarize_pairs
from lingbotvla.recap.evaluation import sha256, write_json
from scripts.recap_finalize_training_ablation import audit, persist, workers_alive


@pytest.fixture
def complete(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    base = {str(i): "unchanged" for i in range(1708)}
    parity = tmp_path / "parity.json"
    write_json(parity, {"base_tensor_sha256": base})
    plan = {
        "parity_report": str(parity),
        "source_sha256": {},
        "evidence_sha256": {},
        "formal_optimizer_seed": SEEDS[0],
        "cells": [],
        "tasks": {t: {"schedules": {str(s): sample_schedule(8, s) for s in SEEDS}} for t in TASKS},
    }
    for t in TASKS:
        diagnostic = {}
        for s in SEEDS:
            draws = [
                {"row": r, "noise_sha256": "shared", "time_sha256": "shared"}
                for row in plan["tasks"][t]["schedules"][str(s)]
                for r in row
            ]
            for b in BACKENDS:
                for i in INITIALIZATIONS:
                    cell = {"task": t, "optimizer_seed": s, "backend": b, "initialization": i}
                    plan["cells"].append(cell)
                    key = f"{b}_s{s}/{i}"
                    folder = root / t / key
                    folder.mkdir(parents=True)
                    path = folder / "recap_adapter.safetensors"
                    save_file(
                        {
                            "model.recap_velocity_lora_a": torch.ones(2, 2, 4),
                            "model.recap_velocity_lora_b": torch.ones(2, 3, 2),
                        },
                        str(path),
                    )
                    write_json(
                        folder / "training.json",
                        {
                            "effective_cell": cell,
                            "steps": 50,
                            "updates": [{"step": x} for x in range(1, 51)],
                            "base_tensors_unchanged": True,
                            "verified_base_tensors": 1708,
                            "cache_loss_exact": True,
                            "cache_gradient_gate_exact": True,
                            "draws": draws,
                            "artifact": {"sha256": sha256(path)},
                        },
                    )
                    diagnostic[key] = [
                        dict(
                            split=split,
                            seed=1,
                            pair_id="p1",
                            time=0.5,
                            noise_seed=4,
                            label=label,
                            base_fm=1.0,
                            positive_fm=1.0 - label * 0.1,
                        )
                        for split in ("train", "holdout")
                        for label in (0, 1)
                    ]
        write_json(root / t / "diagnostic_rows.json", diagnostic)
        write_json(
            root / t / "done.json",
            {
                "cells": 12,
                "optimizer_steps": 600,
                "base_tensor_sha256": base,
                "diagnostics": {k: summarize_pairs(v) for k, v in diagnostic.items()},
            },
        )
    write_json(root / "plan.json", plan)
    write_json(root / "lock.json", {"plan_sha256": sha256(root / "plan.json")})
    return root


def test_complete_reproduces_all_cells_without_success_claim(complete):
    result = audit(complete)
    assert result["training_runs"] == 36 and len(result["cells"]) == 36
    assert result["optimizer_updates"] == 1800 and result["policy_episodes"] == 0
    assert result["formal_optimizer_seed"] == 2101
    assert "not independent" not in result.get("claim", "")


def test_incomplete_or_failed_cohort_not_analyzed(complete):
    (complete / "hanging_mug/done.json").unlink()
    with pytest.raises(ValueError, match="not complete"):
        audit(complete)
    write_json(complete / "stop.json", {"failure": True})
    with pytest.raises(ValueError, match="Incomplete/failed"):
        audit(complete)


def test_changed_artifact_fails_closed(complete):
    p = next(complete.rglob("recap_adapter.safetensors"))
    p.write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash changed"):
        audit(complete)


def test_persistence_conflict_is_not_overwritten(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "x.json").write_text("{}")
    mirror = tmp_path / "mirror"
    inventory = persist(source, mirror)
    assert inventory["x.json"] == sha256(mirror / "x.json")
    (mirror / "x.json").write_text("corrupt")
    with pytest.raises(ValueError, match="conflicts"):
        persist(source, mirror)
    assert (mirror / "x.json").read_text() == "corrupt"


def test_missing_worker_requires_manual_audit(tmp_path):
    write_json(tmp_path / "worker_pids.json", {"mug": {"pid": 123, "start_ticks": "55"}})
    assert not workers_alive(tmp_path, proc=tmp_path / "proc")
    (tmp_path / "mug").mkdir()
    (tmp_path / "mug/done.json").write_text("{}")
    assert workers_alive(tmp_path, proc=tmp_path / "proc")
