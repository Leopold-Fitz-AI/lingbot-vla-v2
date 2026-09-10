from pathlib import Path

import numpy as np
import pytest

from lingbotvla.recap.evaluation import sha256
from scripts.recap_pack_failure_stages import TASK_COUNTS, first_difference, select_pairs, trajectory


def test_complete_blind_pack_keeps_all_discordance_and_fixed_controls():
    arms = {}
    for task, n in TASK_COUNTS.items():
        null = {s: (Path(str(s)), {"success": s % 2 == 0}) for s in range(100)}
        pos = {
            s: (Path(str(s)), {"success": not null[s][1]["success"] if s < n else null[s][1]["success"]})
            for s in range(100)
        }
        arms[task] = {"null": null, "positive": pos}
    chosen = select_pairs(arms)
    assert sum(map(len, chosen.values())) == 100
    for task, n in TASK_COUNTS.items():
        assert set(range(n)) <= set(chosen[task])
        for outcome in (False, True):
            eligible = [s for s in range(n, 100) if arms[task]["null"][s][1]["success"] == outcome]
            assert set(eligible[:5]) <= set(chosen[task])
    assert chosen == select_pairs(arms)
    del arms["hanging_mug"]["null"][99]
    with pytest.raises(ValueError, match="Complete consumed"):
        select_pairs(arms)


def test_npz_inventory_and_exact_difference(tmp_path):
    p = tmp_path / "step.npz"
    np.savez(p, executed_action=np.zeros((50, 14), np.float32), **{"observation::state": np.zeros(14, np.float32)})
    manifest = {"steps": [{"file": p.name, "decision_index": 0}]}
    rows = trajectory(tmp_path / "manifest.json", manifest, {}, {str(p): sha256(p)})
    assert first_difference(rows, rows, "observation") is None
    changed = [{**rows[0], "action": np.ones((50, 14), np.float32)}]
    assert first_difference(rows, changed, "action") == 0
    with pytest.raises(ValueError, match="NPZ changed"):
        trajectory(tmp_path / "manifest.json", manifest, {}, {str(p): "wrong"})
