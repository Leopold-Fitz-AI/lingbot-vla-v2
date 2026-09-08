# Recovery from repeated expert eligibility checks

## Observed failure (not a policy-success claim)

`jy:/dev/shm/recap_confirmatory_20260908_deterministic` stopped on an incomplete
index-42 screen: 540/600 validated episodes; no shortlist, independent
confirmation, or final test. The failed `hanging_mug / positive_d1` job completed
15/20 episodes in each of three bounded whole-task attempts, then rejected
fixed seed 4300019 after six expert checks. This seed had already passed the
outcome-blind preflight and was executed in previous completed policy jobs.
The Null job had earlier encountered similar failures at 4300022 and 4300019.
Its third permitted attempt completed all twenty states; all 32 repeated
completed episodes were byte-identical with matching outcomes.

The error occurs BEFORE requesting a policy action for the rejected seed.
The precise source of expert planner/rollout variability is not established;
do not claim that it is the same MoE defect that was repaired in the policy.

## Demonstrated protocol defect

`deploy/recap_seed_preflight.py` accepts the first N expert-feasible states,
checks a second rollout-style setup, records literal instructions and initial
hashes, and freezes that entire cohort. However,
`experiment/robotwin/eval_policy_client_lingbotvla.py` hard-coded
`expert_check = True`: every policy evaluation reran the expert and could
reject membership again. A frozen seed list is not sufficient to implement
frozen eligibility when this legacy branch is still active.

Two outcome-blind checks support repairing execution rather than substituting
seeds or searching for favorable outcomes:

1. All **564** completed recorded initial observations (24 controls + 540 screen
   episodes) matched the original preflight hashes exactly.
2. Two direct setup-only passes through ALL twenty original `hanging_mug`
   index-42 states, including 4300019/4300022, matched those hashes **40/40**.
   These calls ran neither an expert nor a learned policy.

Evidence: `preflight_initial_hash_audit.json` in the stopped study and
`jy:/dev/shm/recap_locked_cohort_diagnosis_20260908/hanging_mug_s42.json`.

## Repair and safeguards

`deploy/recap_locked_cohort.py` implements `locked-expert-preflight-v1`:

- require an expert-only preflight, SHA-256 map, sibling checksum lock, accepted
  attempt evidence, complete ordered seed list and exact literal instructions;
- do NOT provide a generic skip-expert switch; unverified ordinary evaluations
  retain their legacy expert check;
- execute a verified cohort with zero additional expert rollouts;
- use the exact initial observation checked against preflight for the first
  policy call; mismatches exit 78 and are not retried;
- record actual cohort SHA, initial hashes and expert-call count; the controller
  verifies both metadata and the saved initial NPZ arrays;
- preserve policy failures as failures: no filtering, shortened cohorts or
  success-dependent retries;
- leave all model weights, inference numerics and statistical thresholds alone.

A static audit of all fifty task implementations found three success predicates
that also depend on fields assigned only in `play_once`: `arm_tag` in
`open_laptop`/`place_object_scale`, and `arm_tag` plus `origin_z` in
`put_object_cabinet`. Merely deleting the expert call would break those guards.
The preflight now records an allowlisted evaluator context, and locked execution
restores the native ArmTag and exact scalar type/value. No predicate or tolerance
is rewritten. Other tasks require no copied evaluator fields. Tests explicitly
reject missing/extra context and preserve NumPy scalar dtype bit-for-bit.
On the real simulator, three expert-preflight states per affected task on
consumed index 32 were then initialized twice in fresh environments: **18/18**
matched both preflight images/state and restored evaluator context. All 18
initial predicate calls WITHOUT restoration raised missing-attribute errors;
all worked after restoration. These diagnostics ran no learned policy.

The controller also detects a worker exception as soon as it occurs, publishes
failure status before waiting for other workers, and prevents new job launches.
This fixes the misleading `running` status observed while another task drained.
Failed-job evidence is mirrored even when no complete-cohort `done.json` exists.

## Study versioning

The failed private source, plans, cohorts, outcomes and logs are NOT patched.
A new study will inherit all original index-32 and index-42 cohorts with hash
verification. It must pass the consumed-seed 24-episode A/A control, plus a
complete-trajectory comparison against the prior deterministic Null controls,
before resuming screening. The full 600 screening episodes are rerun under one
protocol, not mixed with favorable previous runs. The same 26 fixed candidates
and four-task independent confirmation family are retained.

Index 42 is already consumed selection data. Indices 43–44 and 50–52 remain
untouched at this amendment. No new optimization, base/adapter file changes,
registry promotion or claim of statistically significant suite improvement.
The [preregistration](recap_confirmatory_protocol.md) remains the analysis contract.
Local regression: **156 passed, 6 skipped**. Remote CPU regression:
**161 passed, 1 CUDA-only module skipped**; physical-GPU-4 kernel checks:
**4 passed**.

## End-to-end gate and persistence

The successor is `jy:/dev/shm/recap_confirmatory_20260908_locked_cohorts`.
Its 24-episode control passed at 15:00 UTC. All twelve paired full trajectories
had byte-identical recorded arrays across independent empty-registry arms.
All twelve Null trajectories also matched the previous live-expert deterministic
control byte-for-byte, including outcomes. See `control.json`,
`control_trajectory_audit.json` and `cross_protocol_control.json`.
Only after these checks may the complete screen rerun proceed.

The failed screen job, all three attempts, frozen cohorts and diagnostic source
are archived in `failed_screen_evidence.tar.gz`, SHA-256:
`2252917f7d901c4a6d6a9727378291170ec860d86e36b6f784c24ec2133b42d2`.
Diagnostics and the archive were read-back verified under
`/vla-cd/ReconVLA/Lingbot-VLA/outputs/recap_cohort_recovery_20260908`;
`persistence.json` lists hashes. Neither this repair nor control success is a
statistical demonstration of 50-task improvement.
