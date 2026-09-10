import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lingbotvla.recap.evaluation import sha256
from scripts.recap_finalize_failure_review import COUNTS, audit
from scripts.recap_pack_full_failure_review import storyboard, validate_rows


def rows(n):
    return [
        {
            "decision": i,
            "observation": {
                "observation::observation.state": np.zeros(14, np.float32),
                "observation::observation.images.cam_high": np.full((12, 16, 3), i + 1, np.uint8),
            },
            "action": np.zeros((50, 14), np.float32),
        }
        for i in range(n)
    ]


def test_all_decisions_rendered_and_missing_never_repeated(tmp_path):
    p = tmp_path / "panel.png"
    storyboard(p, "test", rows(7), rows(1))
    with Image.open(p) as image:
        assert image.size == (6 * 192, 30 + 2 * 318)
        # A d6 is rendered, not selected away. B d6 is grey, not B d0.
        assert image.getpixel((80, 30 + 318 + 80)) == (7, 7, 7)
        assert image.getpixel((80, 30 + 318 + 159 + 120)) == (221, 221, 221)


@pytest.mark.parametrize("defect", ["gap", "nan", "shape", "empty"])
def test_reject_invalid_visual_observations(defect):
    data = rows(2)
    if defect == "gap":
        data[1]["decision"] = 3
    elif defect == "nan":
        data[0]["observation"]["observation::observation.state"][0] = np.nan
    elif defect == "shape":
        data[0]["observation"]["observation::observation.state"] = np.zeros(13)
    else:
        data = []
    with pytest.raises(ValueError):
        validate_rows(data)


@pytest.fixture
def review_pack(tmp_path):
    review = {"review_complete": True}
    for key in ("reviewer", "scope", "confidence_policy", "frame_reference_policy"):
        review[key] = "test qualification"
    public, files = {}, {}
    for task, n in COUNTS.items():
        (tmp_path / task).mkdir()
        review[task] = []
        for i in range(n):
            case = f"{task}/case_{i:03d}"
            review[task].append({"id": i, "A": "unknown", "B": "unknown", "frames": [0], "note": "unobserved"})
            public[case] = {"first_action_difference_decision": 0, "first_observation_difference_decision": 1}
            for suffix in (".png", "_actions.png"):
                p = tmp_path / (case + suffix)
                p.write_bytes(b"test evidence")
                files[case + suffix] = sha256(p)
    for name, value in (
        ("review_template.json", public),
        ("inventory.json", {"pairs": 100, "discordant_pairs": 70, "control_pairs": 30}),
    ):
        p = tmp_path / name
        p.write_text(json.dumps(value))
        files[name] = sha256(p)
    (tmp_path / "full_review_inventory.json").write_text(
        json.dumps({"all_decisions_rendered": True, "missing_frames_repeated": False, "files_sha256": files})
    )
    rp = tmp_path / "review.json"
    rp.write_text(json.dumps(review))
    return rp, tmp_path, review


def test_complete_review_no_private_map_access(review_pack, monkeypatch):
    rp, pack, _ = review_pack
    original = Path.open

    def guarded(self, *args, **kwargs):
        assert "unblind" not in str(self)
        assert "/evaluations/" not in str(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    result = audit(rp, pack)
    assert result["pairs"] == 100 and result["private_mapping_read"] is False
    assert sum(c["unknown"] for c in result["stage_counts"].values()) == 200
    assert len(result["evidence_files_sha256"]) == 202
    assert not result["cases"]["hanging_mug/case_000"]["numeric_action_review_completed"]


@pytest.mark.parametrize("defect", ["incomplete", "duplicate", "unreviewed", "frames", "note", "tamper"])
def test_review_seal_fails_closed(review_pack, defect):
    rp, pack, original = review_pack
    review = copy.deepcopy(original)
    row = review["hanging_mug"][0]
    if defect == "incomplete":
        review["review_complete"] = False
    elif defect == "duplicate":
        review["hanging_mug"][1]["id"] = 0
    elif defect == "unreviewed":
        row["A"] = "unreviewed"
    elif defect == "frames":
        row["frames"] = [0, -1]
    elif defect == "note":
        row["note"] = ""
    else:
        (pack / "hanging_mug/case_000.png").write_bytes(b"changed")
    rp.write_text(json.dumps(review))
    with pytest.raises(ValueError):
        audit(rp, pack)
