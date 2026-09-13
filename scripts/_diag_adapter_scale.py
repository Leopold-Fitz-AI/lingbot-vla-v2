#!/usr/bin/env python3
"""TEMP diagnostic: compare effective velocity-LoRA magnitudes across adapters.

Effective velocity residual for condition row r is (scale/rank) * B[r] @ A[r]
applied to the action-expert hidden state. Reports per-row norms and the
singular values of the effective delta matrix. CPU-only safetensors reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

A_KEY = "model.recap_velocity_lora_a"
B_KEY = "model.recap_velocity_lora_b"


def adapter_stats(path: str, *, scale: float, real_action_dims: int) -> dict:
    with safe_open(path, framework="pt", device="cpu") as archive:
        keys = set(archive.keys())
        if A_KEY not in keys or B_KEY not in keys:
            raise ValueError(f"{path} lacks velocity-LoRA tensors; has {sorted(keys)}")
        a = archive.get_tensor(A_KEY).float()  # [2, rank, hidden]
        b = archive.get_tensor(B_KEY).float()  # [2, action_dim, rank]
    rank = a.shape[1]
    factor = float(scale) / float(rank)
    rows = {}
    for row, name in ((0, "negative_row"), (1, "positive_row")):
        w = factor * (b[row] @ a[row])  # [action_dim, hidden]
        singular = torch.linalg.svdvals(w)
        rows[name] = {
            "norm_A": float(a[row].norm()),
            "norm_B": float(b[row].norm()),
            "effective_frobenius": float(w.norm()),
            "effective_frobenius_real_action_dims": float(w[:real_action_dims].norm()),
            "effective_frobenius_padding_dims": float(w[real_action_dims:].norm()),
            "top_singular_values": [float(v) for v in singular[:8]],
            "effective_rank_99pct": int(
                torch.searchsorted(torch.cumsum(singular**2, 0), 0.99 * (singular**2).sum()) + 1
            ),
        }
    return {
        "path": str(path),
        "scale": float(scale),
        "rank": int(rank),
        "hidden_size": int(a.shape[2]),
        "action_dim": int(b.shape[1]),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", action="append", required=True)
    parser.add_argument("--scale", type=float, default=8.0)
    parser.add_argument("--real-action-dims", type=int, default=14)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = [adapter_stats(path, scale=args.scale, real_action_dims=args.real_action_dims) for path in args.adapter]
    reference = report[0]["rows"]["positive_row"]["effective_frobenius"]
    for entry in report:
        for row in entry["rows"].values():
            row["frobenius_vs_first_adapter"] = row["effective_frobenius"] / reference
    text = json.dumps({"adapters": report}, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
