# Null reproducibility gate: two additional implementation defects

## Scope and status

This is a diagnostic on **already-consumed seed index 32**, not fresh validation
or a 50-task success claim. The first confirmatory study stopped correctly before
opening indices 42–44 or 50–52. Its immutable source, protocol and 24 complete
rollouts remain under `jy:/dev/shm/recap_confirmatory_20260908`, mirrored under
`/vla-cd/ReconVLA/Lingbot-VLA/outputs/recap_confirmatory_20260908`.

This supplements, rather than replaces, the independently demonstrated language,
prefix-noise and activation-window problems in the [earlier audit](recap_root_cause_20260908.md).
Do not attribute every historical score change to this new finding.

## 1. Empty-registry A/A control failed

Both arms used an empty adapter registry, identical base weights, the same GPU
per task, literal instruction maps and per-episode common action noise. Runtime
telemetry confirmed effective Null at EVERY decision, even when the client
requested Positive.

| Task | Null | Requested Positive / effective Null |
|---|---:|---:|
| click_bell | 3/3 | 3/3 |
| hanging_mug | 1/3 | 1/3 |
| place_can_basket | 2/3 | 3/3 |
| stack_bowls_three | 2/3 | 3/3 |
| Total | 8/12 | 10/12 |

The two discordant outcomes are **not RECAP improvements**. All twelve pairs
had zero initial state/image difference, but first predicted actions differed
by approximately 1.2e-6 to 4.9e-6 RMS (maximum coordinate error up to 9.3e-5).
The first observation divergence occurred at decision 1. `take_action` copies
its input array, so this is not recorder mutation of a shared action buffer.
Evidence: `control_divergence_audit.json` and checksummed per-job manifests/NPZs.

## 2. Uninitialized adapter parameters make Null unsafe

`FlowMatchingV2` allocated the down projection with `torch.empty`. Training can
explicitly call `reset_recap_adapter`, but the constructor did not. An external
registry permits absent RECAP tensors in the official base checkpoint; an empty
registry therefore left A uninitialized while B was zero. A replay process
observed an A magnitude of 3.03e35; another fixed-observation replay produced NaN
actions. Multiplication by a null mask does not repair `0 * NaN` or intermediate
overflow.

Fix: initialize enabled adapters in the constructor, using the existing
RNG-independent sine A / zero B initialization. Loading a trained artifact still
overwrites that initialization. Also reject nonfinite artifact tensors and
nonfinite policy outputs before sending actions to the simulator. A poisoned-
allocation unit test exercises the actual constructor block without allocating
the full backbone. Official base files and existing adapter files are untouched.

All 24 original control rollouts had finite recorded actions; the NaN was found
in a separate replay process. Do not attribute the two control outcome changes
solely to initialization. This does not imply that adapters trained with an
explicit reset or loaded from a complete artifact had uninitialized tensors.

## 3. Custom MoE ignored deterministic algorithms

After explicitly resetting A/B, fixed-observation replay still failed:
12 states × 3 identical-input calls in each of three modes (108 calls total):

- default: all 12 states had differing repeat outputs;
- `torch.use_deterministic_algorithms(True)`: all 12 still differed;
- strict mode plus SDPA math-only: all 12 still differed.

The inference-only `lingbotvla/ops/robby_moe.py` packed routes in scheduling-
dependent order and merged experts using floating-point `tl.atomic_add`.
For top-k=4, accumulation order can change low bits. These custom kernels do
not automatically obey PyTorch's deterministic switch. This explains why simply
seeding Torch or disabling attention optimizations was insufficient.

Fix: when deterministic algorithms are enabled, use stable per-expert route
packing, write each route to its own FP32 output buffer, and sum routes in a
fixed routing-rank order. Legacy mode retains its original path. Expert weights,
routing rule, intermediate dtype and GEMMs are unchanged. Strict inference
refuses the old BF16 fallback if the custom kernel fails.

CUDA tests cover FP32/BF16 reference agreement, poisoned/reused workspace,
production dimensions (51 tokens, width 768, 32 experts, top-k 4), fresh
allocations, and CUDA graph replay. All four tests passed on GPU 4.

With the patched kernel, the full-model repeat experiment obtained identical
outputs on **12/12 states in strict mode**, while legacy mode remained
nonrepeatable. Strict SDPA math-only also reproduced all twelve states. These
are fixed-observation inference checks, not simulator success/generalization
results. Evidence: `jy:/dev/shm/recap_null_replay_20260908/run{1,2,3}` and the
associated diagnostic scripts/logs.

## 4. Corrected end-to-end control passed

A new snapshot, `jy:/dev/shm/recap_confirmatory_20260908_deterministic`, preserved
the failed original study and reused its ENTIRE expert-only index-32 cohort.
No policy-success filtering, changed tolerances or fresh validation states were
used to select the numerical fix. `startup_amendment.json` records the provenance.

The 24 new simulator episodes completed. Both independent empty-registry arms
were **9/12**: click_bell 3/3, hanging_mug 1/3, place_can_basket 3/3 and
stack_bowls_three 2/3. All twelve full trajectories matched byte-for-byte in
EVERY recorded array: observations, generated actions and executed actions.
This covered 108 paired policy decisions, including the failed trajectories,
with equal episode step limits. Actual telemetry was strict, non-warn-only and
Null throughout. Evidence: `control.json` and `control_divergence_audit.json` in
the new study. This establishes the repair on these control cases, not a
universal cross-device determinism guarantee.

Tests on the corrected source: local **121 passed, 6 skipped**; remote CPU
**126 passed, 1 CUDA module skipped**; explicit physical-GPU-4 CUDA run
**4 passed**. Neither official base files nor existing trained adapter files
were changed or promoted.

The frozen controller was resumed at 08:44 UTC on 2026-09-08 to execute the
unchanged [selection/confirmation/final plan](recap_confirmatory_protocol.md).
Index 42 may now be used for the predeclared screening only. Indices 43–44 await
the shortlist freeze, and final indices 50–52 await independent confirmation.
No 50-task improvement or significance has yet been demonstrated.

The failed control, environment provenance, direct-replay arrays (including
NaNs), scripts and logs have additional read-back-verified archival at
`/vla-cd/ReconVLA/Lingbot-VLA/outputs/recap_confirmatory_20260908/`:
`null_reproducibility_diagnosis.tar.gz`, SHA-256
`4400055be3d1389d6a8622bb9318763884367a37a7498732e807aedbbfea203a`.
The initial failed study's source/weights/protocol/status remain unchanged.
