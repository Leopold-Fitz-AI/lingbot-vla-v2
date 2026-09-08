# Confirmatory study runbook

## Studies on `ssh jy`

- `W=/dev/shm/recap_confirmatory_20260908_deterministic`: current strict study.
  Control passed (12/12 byte-identical paired trajectories); the full controller
  resumed at 08:44 UTC, 2026-09-08, under tmux `recap-confirmatory-strict`.
  Controller log: `/dev/shm/recap-strict-full-controller.log`.
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
W=/dev/shm/recap_confirmatory_20260908_deterministic
/dev/shm/conda-lingbot-recap/bin/python \
  /dev/shm/lingbot-vla-v2-recap/scripts/recap_study_status.py "$W"
```

`status.json` is a stage snapshot; `final_opened.json` is the authoritative
marker that the final phase has been opened. `done.json` means a job's planned
manifests, telemetry and artifact hashes passed checks. A running job can have
partial manifests that have not yet been validated. Do not infer completion or
significance from an existing result directory or a GPU becoming idle.

## Safe continuation

The initial strict invocation intentionally uses `--stop-after control` so the
patched end-to-end launcher can be reviewed before validation is opened. Once
that control passes and no controller still holds the lock, continue **the same
private snapshot** without `--initialize` or `--stop-after`:

```bash
W=/dev/shm/recap_confirmatory_20260908_deterministic
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

See `plan.json`, `power.json`, `startup_amendment.json` and the archived
preregistration for the exact criteria and assumptions. A +20 pp change on one
of fifty tasks contributes only +0.4 pp to the fixed-suite macro mean. Report
all tasks and adapted-task contributions, not just the winning task.

Authoritative rollouts are on `/dev/shm`. Persistent copies use read-back
verification, but shared NFS has a history of later zero-filled files: verify
checksums before using a mirror. `frozen_inputs.tar.gz` preserves the precise
source, wrapper YAML and compact artifacts used by the study.

Validation and final outcomes remain unproven until those gates actually finish.
Launching the controller is not completing the requested statistical proof.
