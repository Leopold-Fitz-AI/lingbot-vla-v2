#!/usr/bin/env python3
"""Seal a complete coarse visual review without reading the private outcome map.

This validates coverage and evidence integrity, not the correctness of visual
judgments. It never opens policy trajectories, final evaluations, or simulators.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from lingbotvla.recap.evaluation import sha256, write_json


COUNTS = {"hanging_mug": 20, "place_can_basket": 41, "stack_bowls_three": 39}
LABELS = {
    "hanging_mug": {"unknown", "regrasp_transport", "hang_release"},
    "place_can_basket": {"unknown", "can_grasp_transport", "can_placement", "basket_lift"},
    "stack_bowls_three": {"unknown", "bowl_grasp_transport", "base_placement", "stack_placement"},
}


def read_json(path):
    return json.loads(path.read_text())


def audit(review_path, pack):
    review = read_json(review_path)
    if review.get("review_complete") is not True:
        raise ValueError("Complete visual review required; unknown is allowed, unreviewed is not")
    for key in ("reviewer", "scope", "confidence_policy", "frame_reference_policy"):
        if not isinstance(review.get(key), str) or not review[key].strip():
            raise ValueError(f"Missing review qualification: {key}")
    inventory_path = pack / "full_review_inventory.json"
    inventory = read_json(inventory_path)
    if inventory.get("all_decisions_rendered") is not True or inventory.get("missing_frames_repeated") is not False:
        raise ValueError("All-decision non-repeated evidence required")
    expected = inventory["files_sha256"]
    verified = {}

    def verify(relative):
        # Only fixed public paths constructed by this function, never paths from
        # annotations or the inventory. In particular: no unblind_mapping.json.
        path = pack / relative
        actual = sha256(path)
        if actual != expected.get(relative):
            raise ValueError(f"Changed review evidence: {relative}")
        verified[relative] = actual
        return path

    public = read_json(verify("review_template.json"))
    source_inventory = read_json(verify("inventory.json"))
    if (
        source_inventory.get("pairs"),
        source_inventory.get("discordant_pairs"),
        source_inventory.get("control_pairs"),
    ) != (100, 70, 30):
        raise ValueError("Complete fixed consumed-confirmation pack required")
    cases, counts = {}, {}
    for task, count in COUNTS.items():
        rows = review.get(task)
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError(f"Incomplete review: {task}")
        ids = [row.get("id") for row in rows]
        if any(type(index) is not int for index in ids) or ids != list(range(count)):
            raise ValueError(f"Review ids must be complete, unique and ordered: {task}")
        counts[task] = Counter()
        for row in rows:
            case = f"{task}/case_{row['id']:03d}"
            if case not in public:
                raise ValueError(f"Missing public case: {case}")
            frames = row.get("frames")
            if (
                not isinstance(frames, list)
                or not frames
                or any(type(f) is not int or f < 0 for f in frames)
                or frames != sorted(set(frames))
            ):
                raise ValueError(f"Invalid highlighted pair decision indices: {case}")
            if not isinstance(row.get("note"), str) or not row["note"].strip():
                raise ValueError(f"Evidence note required: {case}")
            stages = {}
            for arm in ("A", "B"):
                label = row.get(arm)
                if label not in LABELS[task]:
                    raise ValueError(f"Invalid or unfinished phase label: {case}/{arm}")
                stages[arm] = {
                    "stage": label,
                    "confidence": "not_assessable" if label == "unknown" else "moderate_coarse_phase_only",
                }
                counts[task][label] += 1
            for suffix in (".png", "_actions.png"):
                verify(case + suffix)
            cases[case] = {
                "stages": stages,
                "highlighted_pair_decisions": frames,
                "note": row["note"],
                "first_action_difference_decision": public[case]["first_action_difference_decision"],
                "first_observation_difference_decision": public[case]["first_observation_difference_decision"],
                "storyboard": case + ".png",
                "command_and_proprioception_plot": case + "_actions.png",
                "numeric_action_review_completed": False,
                "causal_mechanism_established": False,
            }
    if set(cases) != set(public):
        raise ValueError("Public pack coverage differs from fixed review")
    return {
        "schema_version": 1,
        "coarse_visual_review_complete": True,
        "pairs": len(cases),
        "review_sha256": sha256(review_path),
        "full_review_inventory_sha256": sha256(inventory_path),
        "source_inventory_sha256": verified["inventory.json"],
        "compiler_sha256": sha256(Path(__file__)),
        "reviewer": review["reviewer"],
        "scope": review["scope"],
        "confidence_policy": review["confidence_policy"],
        "frame_reference_policy": review["frame_reference_policy"],
        "blinding": "Explicit condition/outcome map not read; trajectory lengths are visible",
        "private_mapping_read": False,
        "counts_unit": "Arm trajectories in discordance-enriched review sample, NOT population failure rates",
        "stage_counts": {t: dict(sorted(c.items())) for t, c in counts.items()},
        "evidence_files_sha256": verified,
        "limitations": [
            "Single AI reviewer; no independent human agreement or calibrated confidence",
            "Pre-chunk frames only; no continuous contact/object trajectories or guaranteed terminal observation",
            "Unknown includes visually normal prefixes, occlusion and unresolved metric predicates",
            "Highlighted indices refer to paired panels; missing grey frames are not physical evidence",
            "All numeric action plots preserved and hashed, not individually interpreted",
            "Phase localization and first numerical divergence do not establish intervention causality",
            "No repaired-policy efficacy, training labels, promotion or pilot launch follows from this audit",
        ],
        "cases": cases,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Never overwrite a sealed review audit")
    result = audit(args.review, args.pack)
    write_json(args.out, result, immutable=True)
    print(json.dumps({"pairs": result["pairs"], "stage_counts": result["stage_counts"], "policy_episodes": 0}))


if __name__ == "__main__":
    main()
