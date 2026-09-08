# 50-task confirmatory study — preregistration, 2026-09-08

## Amendment before validation — deterministic implementation

The first startup study (`recap_confirmatory_20260908`) stopped at its consumed-
seed control: empty-registry Null was 8/12 and empty-registry requested Positive
was 10/12, despite actual Null routing in BOTH. No index-42–44 or index-50–52
cohort was opened. Keep that failed study and its original preregistration.

The successor study fixes two implementation defects before starting validation:
initialize absent adapter tensors safely, and enable strict deterministic
algorithms with stable MoE route packing and fixed-order route reduction. The
previous custom inference kernel used floating-point atomics and ignored the
PyTorch deterministic flag. Refuse nonfinite weights/actions and refuse a
backend/precision fallback if the strict kernel fails. Repeat the same consumed-
seed control before proceeding. Candidates, statistical thresholds, sample
sizes and reserved validation/final indices are unchanged. This is an explicit
numerical-protocol amendment, not a change chosen using validation/test outcomes.
See [the reproducibility audit](recap_null_reproducibility_20260908.md).

## Amendment before independent confirmation — locked eligibility

The deterministic successor stopped during index-42 screening: 540/600 planned
policy episodes validated, but `hanging_mug / positive_d1` exhausted three whole-
task attempts at fixed seed 4300019 (six expert rechecks per attempt). Preserve
that study. The client was reopening expert eligibility even after the entire
expert-only cohort had been accepted and checksum-frozen. This is not a failed
learned-policy outcome and does not justify removing the seed.

The new `locked-expert-preflight-v1` execution mode accepts ONLY a checksum-
verified preflight report and sibling lock, matching the entire ordered seed
list and literal instruction map. Expert feasibility remains mandatory ONCE
in outcome-blind preflight; policy execution no longer re-screens membership.
Before any policy call, each initial observation must additionally match the
original preflight hashes exactly. Mismatch exits non-retryably (code 78), never
resampling until a match. Physical setup failures retain the existing bounded
retry policy. Ordinary evaluations without verified cohort evidence retain
their live expert checks. Record/audit actual expert-call count (zero), cohort
SHA and initial hashes, including the recorded NPZ arrays. Preserve evaluator-
only context recorded by preflight: native `arm_tag` for `open_laptop`,
`place_object_scale` and `put_object_cabinet`, plus the latter's dtype-preserved
`origin_z`. These fields are assigned by the original expert and read by the
unchanged success predicates. Restore only this allowlist; do not rerun or
rewrite a success predicate. Missing context is fatal.

Create a NEW frozen study. Inherit ALL original index-32 and index-42 cohorts,
including the problematic seeds, without substituting, filtering or regenerating
them. Repeat the index-32 A/A control and verify complete trajectories against
the previous deterministic control before continuing. Rerun the ENTIRE 600-
episode screen: no previous policy outcomes are imported or selectively reused.
Index 42 is already-consumed selection data, NOT new confirmation evidence.
Candidates/weights/windows/priority, policy numerics, sample sizes, the original
paired-state tolerances and statistical criteria remain unchanged. Indices
43–44 and 50–52 remain unopened until the same shortlist/admission gates pass.
This transparent infrastructure amendment changes neither the models nor the
failure threshold to rescue a favorable score. See [the recovery audit](recap_cohort_recovery_20260908.md).

## Claim and scope

Primary estimand: change in the **equal-weight mean success rate of the fixed
50-task RoboTwin demo_clean suite**, not improvement on every task and not
transfer to a population of unseen tasks. Historical E0 was 4496/5000 (89.92%);
it is inventory only, never the comparator for the new claim. Both policies are
measured under the same new protocol. Report percentage-point uplift, all 50
per-task results, and the number/contribution of adapted tasks.

Statistical success is NOT guaranteed. Do not change thresholds, sample sizes,
registry, exclusions, or analysis after observing final outcomes. A negative or
inconclusive result is a valid study result, not a reason to tune on final seeds.

## Frozen candidates and optimization budget

No additional optimization in this study. Existing artifacts only:

- click_bell: v22 and V2 positive-only/signed/regularized;
- hanging_mug, place_can_basket, stack_bowls_three: V2
  positive-only/signed/regularized each.

All 13 artifacts are copied and SHA-256 frozen, with the 1708-base-tensor export
verification sidecars. Each is screened with all-decision and decision-1-only
windows (26 candidates). There is no new rank/LR/scale/step search. Other tasks
use exactly null/base; never silently apply one task's adapter elsewhere.

## Protocol and eligibility

FP32 model inputs/weights, eager graph, strict deterministic algorithms including
fixed-order custom MoE reduction, chunk length 50, CFG 1, policy seed 900, continuation seed 300,
counterfactual index 0, per-episode common noise, literal frozen instruction
maps generated by task-seed-v2-fixed-candidates32. Baseline and RECAP use the
same adapter-enabled base architecture and normalization. Freeze source code,
wrapper YAML, artifacts, task list and task-to-GPU assignment (GPUs 4–7 only).
Server-reported effective condition, artifact hash, action-noise seed and window
must be recorded and audited; client declarations alone are insufficient.
Set PYTHONHASHSEED=0, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1,
CUBLAS_WORKSPACE_CONFIG=:4096:8 and NVIDIA_TF32_OVERRIDE=0 for all conditions.
These are an explicit new evaluation protocol, not a claim of matching the
historical random-instruction/compiled benchmark protocol. The custom expert
GEMMs and their existing Triton dot precision are otherwise unchanged; FP32 here
describes tensor dtype, not a new all-IEEE-FP32 implementation of every GEMM.

Construct each cohort **without running the learned policy**. Starting at
100000*(seed_index+1), take the first N seeds passing expert feasibility and a
second rollout-style setup. At most 20*N candidate seeds, at most 3 attempts
per candidate. Record every rejection. Never use VLA success to select seeds.
Evaluation gets an exact seed/instruction map plus a verified cohort manifest;
it executes the locked membership rather than rechecking expert feasibility.
There are at most 6 same-seed physical-setup attempts and the launcher's bounded
whole-task retries (maximum three). Frozen-input or initial-hash violations
are fatal and never retried. No substituted
seeds, dropped tasks, shortened cohorts, or success-dependent retries.

Initial positive/null observations must match within state max-abs <= 0.001
and per-image MAE <= 2.0. These thresholds are frozen, not widened after failure.
All rollout manifests and arrays are authoritative on local /dev/shm. Stage and
read-back-verify persistent copies. Infrastructure failures stop the study;
resumption may only complete the same frozen cohorts, never tune the model.

## Stage 0 — protocol control, consumed index 32

Four candidate tasks, 3 episodes each: empty-registry Null and empty-registry
requested Positive. Both MUST report effective Null at every decision and
identical success outcomes. This tests fallback routing and paired noise before
consuming validation seeds. It is not statistical performance evidence.

## Stage 1 — model selection, index 42

20 episodes per candidate task, 20 per candidate/window and one shared Null.
Total: 600 policy episodes. Choose exactly one candidate per task, maximizing
paired net improvement, then minimizing regressions, then a frozen candidate
priority. Ties/ceilings do NOT eliminate a task at this small screening stage.
Priority: click_bell v22-all first; otherwise positive-all, signed-all,
regularized-all, then decision-1 variants in the same artifact order.

Freeze this four-task shortlist before opening indices 43 or 44. Never substitute
another candidate based on confirmation results.

## Stage 2 — independent validation, indices 43–44

50 episodes/task/index, i.e. 100 independent states per shortlisted task.
Evaluate Positive, Null and signed Negative (1200 episodes maximum).
A task is admitted to the final **study** registry only if:

1. complete paired cohorts and all protocol checks pass;
2. Positive > Null > Negative in aggregate;
3. Positive–Null >= 5 percentage points;
4. the two-sided exact paired McNemar/binomial p-value survives Holm correction
   at family-wise alpha=0.05 over the FOUR predeclared tasks.

All four hypotheses remain in the Holm family, including unsuccessful ones.
Do not combine index-42 selection outcomes with confirmation outcomes. If no
candidate passes, stop with `no_validated_candidate`; final seeds stay unused.
No production registry is changed automatically.

## Stage 3 — untouched final evaluation, indices 50–52

Only after shortlist validation and registry SHA-256 freeze:

- ALL 50 tasks, 20 episodes/task/index = 60/task = 3000/condition;
- full independent runs of Null and the frozen task-routed RECAP policy:
  **6000 policy episodes** (do not copy/synthesize baseline outcomes for guards);
- signed Negative on registry tasks only, as a secondary diagnostic;
- use the same GPU for a task across conditions. Alternate temporal condition
  order across indices to limit systematic time-order effects;
- do not compute/read interim final success aggregates to make decisions.

Primary final success requires ALL of:

- 50 complete paired task cohorts, 60 episodes/task/condition;
- equal-weight mean difference > 0;
- two-sided exact paired McNemar/binomial p < 0.05;
- lower bound of a 95% **paired task×seed-index-stratified bootstrap** CI > 0
  (20,000 replicates, RNG seed 20260908);
- no material unadapted-task regression: each unadapted task delta >= -5 pp,
  and the unadapted subset's equal-weight mean delta >= -0.2 pp;
- frozen registry/code/protocol checksums unchanged and no routing violations.

The task list is fixed, not a random sample of tasks. The primary sampling
unit is an independent task/environment episode, NOT action timestep or repeated
branch. Per-task final p-values receive Holm correction over ALL 50 tasks.
No task-level improvement or significance is implied for missing/unchanged
registry tasks. Negative is a secondary mechanism diagnostic, not a basis for
changing the final primary endpoint or choosing another registry.

## Power and runtime

Final N is fixed before validation outcomes: 60 paired states/task. An adapted
task with 15–20 pp uplift and few regressions can make the fixed-suite macro
improvement detectable, even if its contribution is only 0.3–0.4 pp. This is
NOT a claim of +15–20 pp on the suite. Numeric power sensitivity is saved in
`power.json`; it must show assumptions including unadapted-task discordance.
Small real effects or simulator variation may yield an inconclusive result;
do not add final episodes until p crosses 0.05.

Maximum learned-policy budget: 24 controls + 600 screening + 1200 validation +
6000 primary final + 240 negative diagnostic = **8064 episodes**, plus
outcome-blind expert preflight and explicitly logged bounded infrastructure retries.
The full study is an hours-to-days job, supervised under tmux with persistent
stage status and failure reasons. It is not complete when merely launched.
