#!/usr/bin/env python3
"""Condition-blind review pack from COMPLETE, already consumed confirmation only.

All 70 discordant Positive/Null pairs plus 30 seed-ordered tie controls. No final
inputs, no simulator, no automatic grasp/contact labels inferred from success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lingbotvla.recap.evaluation import sha256, write_json


TASK_COUNTS = {"hanging_mug": 10, "place_can_basket": 31, "stack_bowls_three": 29}
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def read_json(path):
    return json.loads(path.read_text())


def select_pairs(arms):
    chosen = {}
    for task, count in TASK_COUNTS.items():
        a, b = arms[task]["positive"], arms[task]["null"]
        if set(a) != set(b) or len(a) != 100:
            raise ValueError("Complete consumed confirmation cohort required")
        discord = [s for s in sorted(a) if a[s][1]["success"] != b[s][1]["success"]]
        if len(discord) != count:
            raise ValueError("Unexpected discordant cohort")
        controls = []
        for success in (False, True):
            eligible = [s for s in sorted(a) if a[s][1]["success"] == b[s][1]["success"] == success]
            if len(eligible) < 5:
                raise ValueError("Incomplete control stratum")
            controls += eligible[:5]
        chosen[task] = sorted(
            discord + controls, key=lambda s: hashlib.sha256(f"phase-review-order-v1/{task}/{s}".encode()).digest()
        )
    return chosen


def trajectory(path, manifest, evidence, expected):
    result = []
    for step in manifest["steps"]:
        p = path.parent / step["file"]
        evidence[str(p)] = sha256(p)
        if evidence[str(p)] != expected[str(p)]:
            raise ValueError("Recorded NPZ changed")
        with np.load(p, allow_pickle=False) as arrays:
            obs = {k: arrays[k].copy() for k in arrays.files if k.startswith("observation::")}
            action = arrays["executed_action"].copy()
            if not obs or action.ndim != 2 or action.shape[1] != 14 or not np.isfinite(action).all():
                raise ValueError("Missing/invalid observations/actions")
            result.append({"decision": step["decision_index"], "observation": obs, "action": action})
    if not result:
        raise ValueError("Empty trajectory")
    return result


def first_difference(left, right, key):
    for a, b in zip(left, right):
        if a["decision"] != b["decision"]:
            raise ValueError("Unaligned decision indices")
        if key == "action":
            x, y = a[key], b[key]
            if x.shape != y.shape or x.tobytes() != y.tobytes():
                return a["decision"]
        else:
            x, y = a[key], b[key]
            if set(x) != set(y):
                raise ValueError("Different observation keys")
            if any(x[k].shape != y[k].shape or x[k].tobytes() != y[k].tobytes() for k in x):
                return a["decision"]
    return None


def storyboard(path, case_id, left, right):
    max_decision = max(len(left), len(right)) - 1
    positions = np.unique(np.linspace(0, max_decision, 6).astype(int)).tolist()
    width = 256 * len(positions)
    image = Image.new("RGB", (width, 660), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (8, 4), case_id + " | A/B are blinded policies; frames are observations BEFORE each action chunk", fill="black"
    )
    for ri, (label, rows) in enumerate((("A", left), ("B", right))):
        for ci, idx in enumerate(positions):
            actual = min(idx, len(rows) - 1)
            frame = rows[actual]
            x, y = ci * 256, 28 + ri * 310
            caption = f"{label} decision {actual}" + (" (last available)" if actual != idx else "")
            draw.text((x + 3, y), caption, fill="black")
            for camera in CAMERAS:
                array = frame["observation"]["observation::observation.images." + camera]
                if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
                    raise ValueError("Unexpected RGB format")
                view = Image.fromarray(array)
                if camera == "cam_high":
                    image.paste(view.resize((256, 192)), (x, y + 20))
                else:
                    xx = x if camera == "cam_left_wrist" else x + 128
                    image.paste(view.resize((128, 96)), (xx, y + 212))
    image.save(path)


def timeline(path, left, right):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b = np.concatenate([r["action"] for r in left]), np.concatenate([r["action"] for r in right])
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    for j, (start, end, label) in enumerate(((0, 6, "Left arm joints"), (7, 13, "Right arm joints"))):
        axes[j].plot(a[:, start:end], alpha=0.6)
        axes[j].plot(b[:, start:end], linestyle="--", alpha=0.6)
        axes[j].set_ylabel(label)
    axes[2].plot(a[:, 6], label="A left")
    axes[2].plot(a[:, 13], label="A right")
    axes[2].plot(b[:, 6], "--", label="B left")
    axes[2].plot(b[:, 13], "--", label="B right")
    axes[2].legend(ncol=4)
    axes[2].set_xlabel("Executed action index (not seconds)")
    fig.suptitle("Solid A / dashed B; raw commands, NOT object poses or contact forces")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pack(study, out):
    if out.exists():
        raise FileExistsError("Never overwrite a review pack")
    arms = {t: {"positive": {}, "null": {}} for t in TASK_COUNTS}
    evidence = {}
    expected = {}
    # Intentionally cannot traverse evaluations/final.
    for path in sorted((study / "evaluations/confirm").glob("*/*/*/job.json")):
        spec = read_json(path)
        if spec["condition"] not in ("positive", "null"):
            continue
        done = path.parent / "done.json"
        if not done.exists():
            raise ValueError("Incomplete confirmation job")
        evidence[str(done)] = sha256(done)
        record = read_json(done)
        files = {r["path"]: r["sha256"] for r in record["files"]}
        expected.update({str(path.parent / rel): h for rel, h in files.items()})
        for item in record["manifests"]:
            p = Path(item["path"])
            if not p.is_absolute():
                p = path.parent / p
            row = read_json(p)
            task = row["metadata"]["task_name"]
            if task not in TASK_COUNTS:
                continue
            seed = int(row["seed"])
            if seed // 100000 - 1 not in (43, 44):
                raise ValueError("Only consumed confirmation indices allowed")
            if sha256(p) != files[str(p.relative_to(path.parent))]:
                raise ValueError("Manifest changed")
            dest = arms[task][spec["condition"]]
            if seed in dest:
                raise ValueError("Duplicate confirmation state")
            dest[seed] = (p, row)
            evidence[str(p)] = sha256(p)
    selected = select_pairs(arms)
    out.mkdir(parents=True)
    public, private = {}, {}
    for task, seeds in selected.items():
        folder = out / task
        folder.mkdir()
        for index, seed in enumerate(seeds):
            case = f"{task}/case_{index:03d}"
            roles = ["positive", "null"]
            if hashlib.sha256(f"phase-blind-v1/{task}/{seed}".encode()).digest()[0] % 2:
                roles.reverse()
            rows = []
            for role in roles:
                p, m = arms[task][role][seed]
                rows.append(trajectory(p, m, evidence, expected))
            storyboard(out / (case + ".png"), case, *rows)
            timeline(out / (case + "_actions.png"), *rows)
            public[case] = {
                "first_action_difference_decision": first_difference(*rows, "action"),
                "first_observation_difference_decision": first_difference(*rows, "observation"),
                "stages": {"A": "unreviewed", "B": "unreviewed"},
                "needs_review": True,
                "allowed_labels": ["approach", "grasp", "carry", "place_or_hang_or_stack", "timeout", "unknown"],
                "evidence_frames": [],
                "note": "No contact forces/object trajectories were recorded; do not invent them",
            }
            private[case] = {
                "task": task,
                "seed": seed,
                "A": roles[0],
                "B": roles[1],
                "outcomes": {role: arms[task][role][seed][1]["success"] for role in roles},
            }
    write_json(out / "review_template.json", public, immutable=True)
    write_json(out / "unblind_mapping.json", private, immutable=True)
    write_json(
        out / "inventory.json",
        {
            "pairs": len(public),
            "discordant_pairs": 70,
            "control_pairs": 30,
            "source_sha256": sha256(Path(__file__)),
            "evidence_sha256": evidence,
            "scope": "Review pack only; stage annotations NOT completed",
        },
        immutable=True,
    )
    return {"pairs": len(public), "scope": "Unreviewed condition-blind evidence"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(pack(args.study, args.out)))


if __name__ == "__main__":
    main()
