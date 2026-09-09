# Confirmatory study completion tracking — 2026-09-09

## Observed milestone (03:10 UTC)

Study: `jy:/dev/shm/recap_confirmatory_20260908_locked_cohorts`.
The private controller remains unchanged (`57f8642` executable provenance).

- Control: **24/24** completed, including the cross-protocol trajectory audit.
- Selection: **600/600** completed on already-consumed index 42.
- Independent confirmation: **1200/1200** completed on indices 43–44.
- The registry was frozen before `final_opened.json` was created.
- All fifty index-50 expert-only cohorts are frozen. The first final Null batch
  has started; **0 primary episodes job-validated, 9 manifests recorded** at
  this timestamp. Counts are not interim success estimates.
- Indices 51–52 are not yet initialized. The frozen controller processes each
  index's preflight, both primary arms and the Negative diagnostic in order.

## Independent confirmation, not the final suite endpoint

| Task | Positive | Null | Negative | Holm p (four tasks) | Admitted |
|---|---:|---:|---:|---:|---|
| click_bell | 100/100 | 84/100 | 53/100 | 0.0001220703125 | yes |
| hanging_mug | 47/100 | 47/100 | 46/100 | 1 | no |
| place_can_basket | 71/100 | 70/100 | 72/100 | 1 | no |
| stack_bowls_three | 80/100 | 79/100 | 88/100 | 1 | no |

Only `click_bell/v22_all` is routed to an adapter in the final study registry;
SHA-256 `b2904a1b9282a73eac958a0406e3a348f50884ff04becf29fd63966ebb6c78ba`.
The other 49 tasks use Null/base and must actually run both policy conditions.
Plan, selection, shortlist, confirmation, registry freeze and final registry
were read-back verified against their NFS mirrors at 03:01 UTC.

The final budget is still **6000 primary episodes + 60 Negative diagnostics**.
A significant single-task confirmation result does NOT prove the final
50-task macro improvement. No final success rates have been inspected here.

## Terminal-only completion observer

`scripts/recap_finalize_study.py` is a separate read-only observer/report tool.
It is NOT copied into or imported by the frozen controller. It cannot launch,
restart or retry a policy job, modify a cohort, change an adapter/registry,
or choose exclusions or new sample sizes.

Run with the inference environment and `PYTHONPATH=STUDY/code`. The observer
verifies that it imports the original frozen `Study` and statistics modules.
Before the terminal `complete` stage it reads only status, cohort lock counts
and job/manifest-path counts; it never reads episode outcomes or final results.
A stopped/dead controller causes an explicit failure report, not a restart.

After complete final evaluation, it:

1. verifies frozen source, weights, wrapper, plan and registry/admission hashes;
2. revalidates every final job using the frozen runtime/cohort/initial-state
   auditor, checks all 50 tasks × 3 indices × 20 pairs and independent guards;
3. checks every final manifest/NPZ checksum locally and against the NFS mirror;
4. recomputes the UNCHANGED frozen statistics and requires exact agreement with
   `final_pairs.json`, `final_result.json` and the controller's terminal claim;
5. writes read-back-verified `reports/completion_audit_v1.json` and
   `reports/completion_report_v1.md`, including all fifty task contributions,
   original CI/exact-test/Holm/guard criteria and secondary Negative results.

This adds reporting/integrity verification, not a post-test analysis amendment.
A negative or nonsignificant complete study is still reported. An incomplete
study never receives a partial-cohort suite estimate. Observer failures are
recorded separately as `reports/completion_observer_failure_v1.json`; they do
not rewrite the frozen study status.

Tests: local **169 passed, 6 skipped**; remote CPU **174 passed, 1 skipped**.
Thirteen observer tests cover no interim reads, incomplete/unpaired strata,
latest-attempt progress counting, terminal negative results, controller-exit
races, no restarts, and correct percentage-point/task-contribution reporting.

The [preregistration](recap_confirmatory_protocol.md) remains the analysis
contract. No production registry promotion is performed.
