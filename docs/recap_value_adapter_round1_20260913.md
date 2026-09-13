# RECAP value-conditioned adapter round-1: null closed-loop result

Date: 2026-09-13. Branch: `recap-causal-policy` (commit `32feb15`).
Server run root: `/dev/shm/recap_value_run_v1/` (persistent copy:
`/vla-cd/ReconVLA/Lingbot-VLA/outputs/recap_value_run_v1/`).

## Question

Does π0.6-style RECAP advantage conditioning (value model labels +
positive/negative-conditioned adapter) improve closed-loop success of the
verified 50k base policy on RoboTwin?

## Pipeline (round-1, all stages completed)

1. Merged multi-source rollouts: 589 episodes, 3353 decisions
   (`rollouts_merged`, `scripts/recap_merge_rollout_runs.py`).
2. VLM embedding cache (`scripts/recap_cache_vlm_embeddings.py`,
   `vlm_embeddings.pt`).
3. Value model training: validation AUROC **0.777** (`recap_vlm_value.pt`).
4. Advantage labeling: **1047 positive / 2306 negative** decisions
   (`rollouts_labeled.jsonl`).
5. Adapter training v1 on all labeled samples,
   `recap_residual_loss_weight=100`.
6. Closed-loop paired evaluation, 6 tasks x 20 seeds, fp32 +
   deterministic + fixed seed map, positive vs null.

## v1 result and root cause

v1 closed loop: positive 98/120 vs null 102/120 (McNemar p=0.424) —
no effect, direction slightly negative.

Diagnosis (`scripts/_diag_fm_loss.py`, `scripts/_diag_adapter_scale.py`,
output in `diagnostics/`):

- The signed axis was **inverted by the negative-sample majority** (2.2:1).
  On positive-label training samples, the positive condition was the worst
  of the three conditions (97.5% agreement).
- `recap_residual_loss_weight=100` compressed the adapter residual norm to
  **0.031** Frobenius, 33% of the historical successful v22 adapter (0.093).

## v2 fix

- Training data: positive-only, 1047 samples
  (`lerobot_recap_positive_only`, `--keep-labels positive`).
- `recap_residual_loss_weight=0`, 1500 steps.
- Export: `recap_adapter_value_v2.safetensors`
  (sha256 `9ec79c1f...`; 1708 base tensors verified unchanged).

Gates:

- **Gate A (scale): passed.** ||W||_F = 1.256 (v1: 0.031, v22: 0.093).
- **Gate B (directionality): threshold missed, direction fixed.** On
  positive-label samples: L(positive)=0.00992 < L(null)=0.01006 <
  L(negative)=0.01370. Win rate vs null 0.55 (preset threshold 0.70; v1 was
  0.025), vs negative 0.975. The margin is small because positive-label
  actions were executed by the behavior policy itself, so the null
  condition already fits them well.

## v2 closed-loop result (final)

120 paired episodes, same protocol as v1; adapter sha256 verified in every
manifest. Both conditions rc=0.

| Task | positive | null | diff |
|---|---:|---:|---:|
| click_bell | 18/20 | 16/20 | +2 |
| move_stapler_pad | 19/20 | 19/20 | 0 |
| place_can_basket | 14/20 | 13/20 | +1 |
| stack_bowls_three | 15/20 | 18/20 | -3 |
| stamp_seal | 19/20 | 18/20 | +1 |
| turn_switch | 18/20 | 18/20 | 0 |
| **Total** | **103/120** | **102/120** | **+1** |

McNemar discordant pairs: b=8 (positive-only wins), c=7 (null-only wins),
exact two-sided **p=1.0**.

## Conclusion

At this scale (589 source episodes, strong 50k base with 85% null success
on the eval suite), advantage-conditioned adapter training produces a real
but tiny in-distribution training signal that does **not** convert into a
measurable closed-loop gain. The v2 repair fixed the inverted direction and
the over-regularization, and the result moved from slightly negative to
exactly zero — a valid null result, not a pipeline failure.

Contributing factors:

- Ceiling: 5/6 tasks already at >=65% base success; positive-label actions
  are behavior-policy actions the base already fits.
- Data scale: 1047 positive samples is small for adapter conditioning.

## If resumed

- Value-guided best-of-N action selection at inference uses the existing
  value model (AUROC 0.777) and is not subject to the behavior-policy
  ceiling; it is the more promising follow-up than round-2 adapter retraining.
- Round-2 adapter retraining on the 240 new rollouts is possible but
  expected marginal given the effect size above.
