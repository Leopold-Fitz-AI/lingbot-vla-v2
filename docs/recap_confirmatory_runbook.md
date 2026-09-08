# Confirmatory study runbook

## Studies on `ssh jy`

- `W=/dev/shm/recap_confirmatory_20260908_locked_cohorts`: current amended study.
  Reuses ALL original control/screen cohorts, not their policy results. Initial
  control launched at 14:45 UTC with `--stop-after control`, tmux
  `recap-locked-control`, log `/dev/shm/recap-locked-controller.log`.
  After both control audits passed, the full frozen controller resumed at
  **15:13 UTC**, tmux `recap-confirmatory-locked`, PID 2101880, log
  `/dev/shm/recap-locked-full-controller.log`. At 15:15 it was running the four
  shared-Null screening jobs (0/600 validated yet); confirmation/final stayed sealed.
- `/dev/shm/recap_confirmatory_20260908_deterministic`: stopped during screening
  at **540/600 validated episodes** when `hanging_mug / positive_d1` exhausted
  all three attempts at seed 4300019. No shortlist/confirmation/final access.
  Preserve its frozen code, failed attempts and outcomes; do NOT resume it.
- `/dev/shm/recap_confirmatory_20260908`: failed initial A/A startup; preserve it.
  Its validation/final cohorts were never opened. Do not overwrite its frozen code.
- `/dev/shm/recap_null_replay_20260908`: fixed-observation diagnosis of custom
  MoE nondeterminism and absent-adapter initialization.
- Persistent mirrors: `/vla-cd/ReconVLA/Lingbot-VLA/outputs/<study basename>`.

Only GPUs 4–7 are authorized. Official base and existing compact artifacts are
read-only inputs, copied/checksummed where appropriate. No production adapter
registry is changed automatically.

## Progress, without peeking at interim final success

```bash
W=/dev/shm/recap_confirmatory_20260908_locked_cohorts
/dev/shm/conda-lingbot-recap/bin/python \
  /dev/shm/lingbot-vla-v2-recap/scripts/recap_study_status.py "$W"
```

`status.json` is a stage snapshot; `final_opened.json` is the authoritative
marker that the final phase has been opened. `done.json` means a job's planned
manifests, telemetry and artifact hashes passed checks. A running job can have
partial manifests that have not yet been validated. Do not infer completion or
significance from an existing result directory or a GPU becoming idle.

## Safe continuation

The amended invocation intentionally uses `--stop-after control`. Require both
`control.json` AND the passed `cross_protocol_control.json` (all twelve complete
Null trajectories match the previous strict study). Once those gates pass and
no controller still holds the lock, continue **the new private snapshot** without
`--initialize` or `--stop-after`. This is NOT permission to retry the stopped
predecessor or to regenerate its cohorts:

```bash
W=/dev/shm/recap_confirmatory_20260908_locked_cohorts
export CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONNOUSERSITE=1 PYTHONPATH="$W/code"
/dev/shm/conda-lingbot-recap/bin/python -u \
  "$W/code/scripts/recap_confirmatory_study.py" --study "$W"
```

Do not launch a second controller while that tmux/process is active. For a
necessary continuation, run under tmux and capture stdout/stderr to a local log. Completed jobs are
reused only after their evidence hashes are verified. The launcher has bounded
same-cohort task retries. A supervisor interruption with incomplete evidence
fails closed: **do not delete launch markers, substitute seeds, or rerun based
on an unfavorable success rate**. Diagnose infrastructure and record any
recovery explicitly. Modifying executable code requires a new audited protocol
version, not bypassing `code_sha256` or rewriting an immutable decision file.

## Gates and artifacts

1. `control.json`: 24 consumed-seed episodes; both empty-registry branches must
   have identical success outcomes and actual Null routing.
2. `screen_selection.json`, `shortlist_lock.json`: 600 index-42 screening
   episodes, four shortlisted candidates, frozen before confirmation.
3. `confirmation.json`: 1200 independent index-43–44 Positive/Null/Negative
   episodes. Holm-corrected four-task tests and directional/practical-effect
   gates decide inclusion. `no_validated_candidate` stops without final access.
4. `registries/final.json`, `registry_freeze.json`: study-only registry freeze.
5. `final_pairs.json`, `final_result.json`: complete 50-task, 60-episode/task
   paired analysis. The primary final budget is 6000 episodes, plus negative
   diagnostics only on registered tasks. There is no final-test-driven tuning.

See `plan.json`, `power.json`, `cohort_inheritance.json` and the archived
preregistration for the exact criteria and assumptions. A +20 pp change on one
of fifty tasks contributes only +0.4 pp to the fixed-suite macro mean. Report
all tasks and adapted-task contributions, not just the winning task.

Authoritative rollouts are on `/dev/shm`. Persistent copies use read-back
verification, but shared NFS has a history of later zero-filled files: verify
checksums before using a mirror. `frozen_inputs.tar.gz` preserves the precise
source, wrapper YAML and compact artifacts used by the study.

## Screening infrastructure event (2026-09-08)

The index-42 `hanging_mug` Null job required all three predeclared whole-task
attempts. Attempts 1/2 completed 17/15 episodes, then failed expert revalidation
of fixed seeds 4300022/4300019 after six setup attempts. Attempt 3 completed all
20 frozen states. No seed was substituted, no fourth attempt was added, and
no candidate/outcome criterion chose the retry. Every one of the 32 repeated
completed episodes had byte-identical arrays and the same outcome in attempt 3.
The two extra incomplete attempts are infrastructure overhead, not additional
independent samples. See `expert_retry_resolution.json` and the read-back-
verified `screen_null_hanging_retry_evidence.tar.gz` in the strict study.
The underlying expert-planner variability has not been localized; any future
failure exhausting the existing retry limit must still stop the study.

The later Positive-d1 job exhausted its cap and the strict study STOPPED at
540/600 validated episodes. A live controller while concurrent tasks drained
was not evidence that screening could complete. No fourth attempt, seed
substitution, changed adapter or dropped candidate is permitted.

The [locked-eligibility repair](recap_cohort_recovery_20260908.md) removes repeat
expert screening ONLY for checksum-verified preflight cohorts, restores any
required evaluator-only context, and fails non-retryably on an initial-state
hash mismatch. The new study reruns the full control and full 600-episode
screen. Index 42 remains consumed selection data; indices 43–44 and 50–52 stay
sealed until the original stage gates permit access.

The new **24/24-episode control passed** (recorded at 15:00 UTC). Both
independent arms produced byte-identical arrays in all twelve paired full
trajectories. All twelve new Null trajectories also matched the previous strict
control byte-for-byte, including outcomes. `control_trajectory_audit.json` and
`cross_protocol_control.json` are read-back verified. `control_audit_roles.json`
clarifies the generic purpose text in the same-study trajectory report; roots,
manifest hashes and `paired_arms_same_study` identify the actual comparison.
The repair is therefore
eligible for a complete screen rerun; no significant 50-task gain is yet proven.
