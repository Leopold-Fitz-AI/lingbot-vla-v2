import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/dev/shm/recap_value_run_v1/rollouts_merged")

# 1) branch composition of the merged training data (from merged_from metadata)
branch_counter = Counter()
branch_outcome = defaultdict(lambda: [0, 0])
for manifest_path in ROOT.rglob("manifest.json"):
    m = json.loads(manifest_path.read_text())
    src = m.get("metadata", {}).get("merged_from", "?")
    # merged_from = <root>/<branch>/<episode>; extract branch component
    parts = Path(src).parts
    branch = parts[-2] if len(parts) >= 2 else "?"
    root_name = [p for p in parts if p.startswith(("recap_", "click_bell", "rollouts"))]
    branch_counter[branch] += 1
    branch_outcome[branch][0 if m.get("success") else 1] += 1

print("=== branch composition (episode count / success / fail) ===")
for branch, n in branch_counter.most_common():
    s, f = branch_outcome[branch]
    print(f"  {branch:<45} n={n:4d}  succ={s:4d} fail={f:4d}")

# 2) label vs outcome crosstab
print("\n=== recap_label x episode outcome (step level) ===")
cross = defaultdict(int)
adv_by_key = defaultdict(list)
for line in open("/dev/shm/recap_value_run_v1/rollouts_labeled.jsonl"):
    ep = json.loads(line)
    for s in ep["steps"]:
        label = s["recap_label"]
        outcome = "succ" if ep["success"] else "fail"
        cross[(label, outcome)] += 1
        adv_by_key[(label, outcome)].append(s["recap_advantage"])
for (label, outcome), n in sorted(cross.items()):
    advs = np.array(adv_by_key[(label, outcome)])
    print(f"  label={label:+d} outcome={outcome}: n={n:5d}  adv mean={advs.mean():8.2f}")

# 3) label-position profile: where in episodes do positive labels concentrate?
print("\n=== positive label rate by relative decision position ===")
early = Counter()
for line in open("/dev/shm/recap_value_run_v1/rollouts_labeled.jsonl"):
    ep = json.loads(line)
    steps = ep["steps"]
    n = len(steps)
    for i, s in enumerate(steps):
        bucket = "first-third" if i < n / 3 else ("mid-third" if i < 2 * n / 3 else "last-third")
        early[(bucket, s["recap_label"])] += 1
for bucket in ("first-third", "mid-third", "last-third"):
    p = early[(bucket, 1)]
    t = sum(early[(bucket, l)] for l in (-1, 0, 1))
    print(f"  {bucket:<12} positive rate = {p}/{t} = {p / t:.2%}")

# 4) per-task value model sanity: within-episode V trend (should increase toward success)
print("\n=== mean value trend within successful episodes (first vs last decision) ===")
for line in open("/dev/shm/recap_value_run_v1/rollouts_labeled.jsonl"):
    pass
first_v, last_v = [], []
for line in open("/dev/shm/recap_value_run_v1/rollouts_labeled.jsonl"):
    ep = json.loads(line)
    if not ep["success"]:
        continue
    steps = ep["steps"]
    if len(steps) >= 3:
        first_v.append(steps[0]["value"])
        last_v.append(steps[-1]["value"])
print(f"  successful eps: V(first)={np.mean(first_v):.1f} -> V(last)={np.mean(last_v):.1f} (n={len(first_v)})")
