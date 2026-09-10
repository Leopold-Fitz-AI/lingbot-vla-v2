#!/usr/bin/env python3
"""All-decision blinded confirmation review; no interpolation of missing frames.

Complements, never overwrites, the sparse v1 three-camera storyboards. Actual
proprioception and commanded actions are separated. Head-camera overviews cannot
establish contact forces, terminal states, or a causal mechanism.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lingbotvla.recap.evaluation import sha256, write_json
from scripts import recap_pack_failure_stages as sparse


def validate_rows(rows):
    if not rows or len(rows) > 256:
        raise ValueError("Invalid or oversized review trajectory")
    for index, row in enumerate(rows):
        if row["decision"] != index:
            raise ValueError("Decisions must be complete and contiguous")
        for array in row["observation"].values():
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite review observation")
        state = row["observation"]["observation::observation.state"]
        if state.shape != (14,):
            raise ValueError("Expected actual 14-dimensional proprioception")


def storyboard(path, case_id, left, right):
    for rows in (left, right):
        validate_rows(rows)
    n = max(len(left), len(right))
    columns = min(6, n)
    width, height = 192, 318
    image = Image.new("RGB", (columns * width, 30 + height * ((n + columns - 1) // columns)), "white")
    draw = ImageDraw.Draw(image)
    draw.text((4, 3), case_id + " | EVERY pre-chunk observation | A above B | head camera only", fill="black")
    for index in range(n):
        x, y = index % columns * width, 30 + index // columns * height
        for ri, (label, rows) in enumerate((("A", left), ("B", right))):
            yy = y + ri * 159
            draw.text((x + 2, yy), f"{label} d{index}", fill="black")
            if index >= len(rows):
                draw.rectangle((x, yy + 15, x + width - 1, yy + 158), fill="#dddddd")
                draw.text((x + 4, yy + 50), "NO RECORDED FRAME", fill="black")
                continue
            array = rows[index]["observation"]["observation::observation.images.cam_high"]
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError("Invalid RGB frame")
            image.paste(Image.fromarray(array).resize((width, 144)), (x, yy + 15))
    image.save(path)


def timeline(path, left, right):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 1, figsize=(13, 12), sharex=True)
    groups = ((list(range(6)), "left joints"), (list(range(7, 13)), "right joints"), ([6, 13], "grippers"))
    for label, rows, style in (("A", left, "-"), ("B", right, "--")):
        validate_rows(rows)
        actions = np.concatenate([r["action"] for r in rows])
        starts = np.cumsum([0] + [len(r["action"]) for r in rows[:-1]])
        states = np.stack([r["observation"]["observation::observation.state"] for r in rows])
        for gi, (indices, title) in enumerate(groups):
            for ci, channel in enumerate(indices):
                color = f"C{ci}"
                axes[gi].plot(actions[:, channel], linestyle=style, color=color, alpha=0.7, label=f"{label}:{channel}")
                axes[gi + 3].plot(
                    starts,
                    states[:, channel],
                    linestyle=style,
                    marker=".",
                    color=color,
                    alpha=0.7,
                    label=f"{label}:{channel}",
                )
            axes[gi].set_ylabel("COMMAND " + title)
            axes[gi + 3].set_ylabel("OBSERVED " + title)
    axes[2].legend(ncol=4)
    axes[5].legend(ncol=4)
    axes[-1].set_xlabel("Executed action index; observed points only at chunk boundaries (lines are visual guides)")
    fig.suptitle("Solid A / dashed B; no object poses, forces, or unrecorded post-terminal state inferred")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = sparse.pack(args.study, args.out, render_storyboard=storyboard, render_timeline=timeline)
    files = {str(f.relative_to(args.out)): sha256(f) for f in sorted(args.out.rglob("*")) if f.is_file()}
    write_json(
        args.out / "full_review_inventory.json",
        {
            "renderer_sha256": sha256(Path(__file__)),
            "selector_sha256": sha256(Path(sparse.__file__)),
            "files_sha256": files,
            "all_decisions_rendered": True,
            "missing_frames_repeated": False,
            "observed_joints_separate_from_commands": True,
            "semantic_review_complete": False,
            "scope": "Blinded consumed confirmation, never final; full HEAD camera and original sparse wrist views complement each other",
        },
        immutable=True,
    )
    print(result)


if __name__ == "__main__":
    main()
