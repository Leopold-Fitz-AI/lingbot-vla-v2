# RECAP Development Guide

This document describes the first RECAP integration for LingBot-VLA 2.0.
RECAP (RL with Experience and Corrections via Advantage-conditioned Policies)
keeps the existing flow-matching policy objective and conditions it on whether
a behavior-policy action has positive or negative advantage.

## Current implementation status

Implemented:

- positive, negative, and null conditions shared by training and deployment;
- a null-preserving velocity-LoRA action adapter with a signed advantage axis;
- exact-decision conditioning and counterfactual action-noise schedules;
- causal paired-decision selection with outcome balancing and observation checks;
- per-sample condition dropout for the unconditional policy branch;
- backward compatibility for legacy expert datasets;
- categorical distributional value-head/loss primitives with 201 bins by default;
- time-to-success rewards, Monte-Carlo returns, advantages, and ternary labels;
- an offline JSONL labeling utility;
- dependency-light RoboTwin rollout recording with exact generated/executed action lengths;
- an independently trainable state+task categorical value baseline and predictor;
- deployment defaults to the positive branch and permits per-observation overrides;
- optional positive-vs-null CFG that combines velocities at every flow denoising step;
- atomic attachment of labeled episode/frame indices to cloned LeRobot Parquet datasets.

Still required before claiming task-suite-wide RECAP:

- a visual-language value encoder, or simulator interventions at every relevant
  decision, to estimate advantage where vision is required;
- repeated counterfactual collection/training rounds across tasks;
- held-out evaluation on all target tasks, not only the `click_bell` diagnosis
  described below.

## Root cause and corrected objective

An episode outcome is **not** an action-advantage label. In the original
pipeline, successful and failed episodes differed in environment state, task
wording, sampled action noise, and all later policy actions. Assigning the final
outcome to every decision therefore trained two behavior-cloning policies on a
confounded partition of the data. This explains why positive and negative
conditions often moved success in the same direction and why results changed
with seed.

The corrected collection protocol holds the environment seed, task text, prefix
actions, and continuation action-noise schedule fixed. It varies exactly one
policy decision. Only groups containing both successful and failed branches are
kept. This is an intervention on the selected action rather than a correlation
between a trajectory and its final outcome.

Failed-action BC is still not a reliable way to define a negative policy: its
mean residual can align with the successful residual. The safe adapter therefore
learns only a useful positive residual and can derive negative as its exact
inverse (`recap_signed_velocity_axis: true`). Null is always the exact frozen
base policy. The resulting policy axis is structurally
`base - residual`, `base`, `base + residual` for negative, null, and positive.

## Label schema

The policy dataset uses a scalar per-frame LeRobot column named `recap_label` by
default:

| Value | Meaning | Policy prompt |
| ---: | --- | --- |
| `1` | Positive advantage | `Advantage: positive.` |
| `0` | Negative advantage | `Advantage: negative.` |
| `-1` | Neutral or unconditional | Original task prompt only |

The raw scalar advantage should be stored separately as `recap_advantage`.
Keeping labels and raw advantages separate prevents a continuous advantage from
being mistaken for a binary policy condition.

When RECAP is enabled, legacy datasets without `recap_label` use
`data.recap_missing_condition`. Its default is `positive`, which treats expert
demonstrations as positive examples. Use `error` while validating a fully
labeled rollout dataset.

## Policy configuration

```yaml
data:
  recap_enabled: true
  recap_indicator_key: recap_label
  recap_missing_condition: positive
  recap_condition_dropout: 0.1
  recap_condition_name: Advantage
```

The condition is prepended before the task so that tokenizer truncation does
not silently remove it. Condition dropout is applied only during training.
`policy_eval=True` never drops the requested condition.

Text prompt conditioning is retained for compatibility, but it should be
turned off for a safe adapter experiment: the official model was never trained
on an `Advantage:` prefix, so even a zero-update positive prompt changes the
base policy. The recommended configuration freezes every official parameter
and trains only a condition-specific residual on the action velocity:

```yaml
data:
  recap_enabled: true
  recap_prompt_enabled: false
  recap_adapter_enabled: true
  recap_condition_dropout: 0.0
  recap_missing_condition: error

train:
  recap_adapter_enabled: true
  recap_adapter_type: velocity_lora
  recap_adapter_rank: 8
  recap_adapter_scale: 8.0
  recap_adapter_init_std: 0.02
  train_recap_adapter_only: true
  reset_recap_adapter: true
  recap_signed_velocity_axis: true
```

The null branch has no trainable row and produces a bitwise-zero residual.
Train the signed configuration with positive causal samples only; negative
samples are useful for identifying mixed-outcome groups but are not a second BC
target. After training, verify every non-RECAP tensor against the base checkpoint.

## RoboTwin rollout collection

The queue-based RoboTwin launcher can record rollout bundles without changing
its default evaluation behavior. Pass an explicit directory, or `auto` to place
bundles under the timestamped evaluation run directory:

```bash
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
  --eval_workdir /path/to/RoboTwin \
  --model_path /path/to/checkpoint \
  --recap_condition positive \
  --recap_rollout_dir auto
```

Every episode has an atomic `manifest.json` and one compressed NPZ per policy
decision. Each NPZ preserves the raw camera/state observation, the complete
generated action chunk, and the exact executed prefix. The latter matters when
the environment succeeds partway through a 50-step action chunk. Incomplete
processes have no final manifest and are therefore ignored by downstream tools.

The launcher synchronizes `deploy/recap_rollout_recorder.py` and the evaluation
client into the RoboTwin checkout before starting simulation workers.

For causal collection, use the same continuation seed in every branch, vary the
policy seed, and select the one decision being tested:

```bash
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
  --eval_workdir /path/to/RoboTwin \
  --model_path /path/to/base-checkpoint \
  --policy_seed 600 \
  --continuation_policy_seed 100 \
  --common_noise_per_episode \
  --counterfactual_policy_decision 1 \
  --recap_rollout_dir /path/to/branch-600
```

Repeat with different `--policy_seed` values while keeping everything else
fixed. Then select only mixed-outcome groups:

```bash
python scripts/recap_select_paired_decisions.py \
  --input /path/to/branch-600 \
  --input /path/to/branch-601 \
  --input /path/to/branch-602 \
  --output /path/to/paired \
  --decision-index 1 \
  --balance-outcomes \
  --pairwise-match \
  --maximum-pairs-per-state 2
```

Exact observation equality is the default. `--pairwise-match` computes a
maximum-cardinality, minimum-drift bipartite matching and retains only
success/failure pairs that individually satisfy the observation tolerances; an
outlier branch can no longer invalidate an otherwise causal pair. Runs
collected on different GPUs can exhibit tiny numerical drift before the branch
point; tolerances may be specified explicitly with `--observation-atol` and
`--image-mae-tolerance`, but their values and per-pair maxima must be recorded
and audited. Prefer sequential collection on one GPU when practical. Broad
task selection must enforce independent-state and sample gates, for example
`--minimum-mixed-states 8 --minimum-positive 16`; repeated actions from one
state are not independent evidence.

## Rollout episode interchange format

`scripts/recap_label_rollouts.py` consumes one JSON object per episode:

```json
{
  "episode_id": "pick-cup/000001",
  "task": "pick up the cup",
  "success": true,
  "steps": [
    {
      "value": -12.4,
      "reward": -1.0,
      "terminated": false,
      "valid": true
    },
    {
      "value": -11.1,
      "reward": 0.0,
      "terminated": true,
      "valid": true
    }
  ]
}
```

`reward` is optional only when it is omitted from every step. In that case the
script constructs the RECAP time-to-success reward from `success`: non-terminal
steps receive `-1`, successful terminals receive `0`, and failed terminals
receive `-failure-penalty`.

Example:

```bash
python scripts/recap_label_rollouts.py \
  --input rollouts_with_values.jsonl \
  --output rollouts_labeled.jsonl \
  --failure-penalty 1000 \
  --gamma 1.0 \
  --neutral-margin 0.5
```

Each output step receives:

```json
{
  "recap_return": -12.0,
  "recap_advantage": 0.4,
  "recap_label": 1
}
```

A task-specific threshold map is supported:

```json
{
  "pick up the cup": 0.25,
  "fold the shirt": 1.0
}
```

```bash
python scripts/recap_label_rollouts.py \
  --input rollouts_with_values.jsonl \
  --output rollouts_labeled.jsonl \
  --threshold-map recap_thresholds.json
```

## Value learning primitives

`lingbotvla.recap.CategoricalValueHead` accepts a pooled context embedding and
outputs categorical value logits. Defaults match the RECAP distributional
recipe:

```python
from lingbotvla.recap import CategoricalValueHead, categorical_value_loss

head = CategoricalValueHead(
    hidden_size=2560,
    num_bins=201,
    value_min=-2000,
    value_max=0,
)
logits = head(context_embedding)
loss = categorical_value_loss(
    logits,
    monte_carlo_return,
    head.value_support,
    valid_mask=valid_mask,
)
value = head.expected_value_from_logits(logits)
```

The value range must cover the configured failure penalty and maximum episode
length. Returns outside the support are clipped to the nearest bin by target
discretization, so clipping frequency must be monitored.

A first runnable baseline uses normalized proprioceptive state plus a learned
RoboTwin task embedding. Train it directly from recorded bundles:

```bash
python tasks/vla/train_recap_value.py \
  --rollout-dir /path/to/recap_rollouts \
  --output output/recap_value.pt \
  --failure-penalty 1000 \
  --value-min -2000 \
  --value-max 0
```

The train/validation split is performed by complete episode, never by frame.
The checkpoint includes state normalization, the task vocabulary, return
settings, and the best distributional model. Generate value-annotated episode
JSONL and then binarize advantages:

```bash
python scripts/recap_predict_values.py \
  --rollout-dir /path/to/recap_rollouts \
  --checkpoint output/recap_value.pt \
  --output output/rollouts_with_values.jsonl

python scripts/recap_label_rollouts.py \
  --input output/rollouts_with_values.jsonl \
  --output output/rollouts_labeled.jsonl \
  --neutral-margin 0.5
```

When rollouts have already been converted into a LeRobot dataset and the JSONL
steps preserve matching `episode_index` and `frame_index`, attach labels to an
atomic clone of that dataset:

```bash
python scripts/recap_attach_lerobot_labels.py \
  --dataset-root datasets/robotwin_rollouts \
  --labels output/rollouts_labeled.jsonl \
  --output-root datasets/robotwin_rollouts_recap \
  --missing error
```

The tool appends an `int8 recap_label` Parquet column and updates
`meta/info.json`. It never edits the source dataset in place. Decision index may
be used as frame index only with the explicit `--use-decision-index` flag, since
silently equating these indices can attach labels to the wrong observations.
Use `--missing positive` only when unlabeled rows are known expert
demonstrations; `--missing error` is the safe rollout default.

The rollout loader reconstructs action-level time-to-success reward from each executed
chunk length: all elapsed actions receive `-1`, successful terminal actions
receive `0`, and failed terminal actions receive `-failure-penalty`. For
`gamma < 1`, both within-chunk and between-chunk discounting use the true
executed duration.

This baseline validates the collection/value/labeling loop, but it cannot infer
visual progress that is absent from joint state. It should not be presented as
the final RECAP value model; the next version must pool the VLM observation and
task representation.

## Deployment

For a RECAP-trained checkpoint, deployment defaults to the positive branch:

```bash
python -m deploy.lingbot_vla_v2_policy \
  --model_path /path/to/checkpoint \
  --recap_condition positive
```

A full checkpoint is not required for a frozen-base adapter. Export it while
verifying every base tensor, then overlay the compact artifact on a base
checkpoint whose inference YAML enables the same adapter architecture:

```bash
python scripts/recap_export_adapter.py \
  --checkpoint /path/to/tuned/hf_ckpt \
  --base-checkpoint /path/to/official/hf_ckpt \
  --output /safe/local/path/recap_adapter.safetensors \
  --signed-velocity-axis

python -m deploy.lingbot_vla_v2_policy \
  --model_path /path/to/base/checkpoint-via-adapter-enabled-wrapper \
  --recap_adapter_path /path/to/recap_adapter.safetensors \
  --recap_condition positive
```

The overlay loader rejects non-RECAP tensors, missing adapter tensors, shape
mismatches, incompatible base keys, and signed-axis metadata/config mismatches.

Supported server defaults are `positive`, `negative`, and `null`. Conditioning
can be restricted to a window with `--recap_condition_start_decision` and
`--recap_condition_decisions`; for example, start `1` and count `1` conditions
only decision index 1. A WebSocket
observation can override the server default by providing the configured
indicator key:

```python
observation["recap_label"] = 1   # positive
observation["recap_label"] = 0   # negative ablation
observation["recap_label"] = -1  # null/unconditional ablation
```

Old checkpoints remain unchanged because their saved data configuration does
not enable RECAP.

A full adapter-only training checkpoint still contains a copy of every frozen
base tensor. Export a compact, verified artifact instead:

```bash
python scripts/recap_export_adapter.py \
  --checkpoint /path/to/trained/hf_ckpt \
  --base-checkpoint /path/to/official/hf_ckpt \
  --output /path/to/recap_adapter.safetensors \
  --signed-velocity-axis
```

The exporter fails if any non-RECAP tensor differs elementwise from the base.
Deploy the artifact with a base-checkpoint wrapper whose inference YAML enables
the same adapter architecture:

```bash
python -m deploy.lingbot_vla_v2_policy \
  --model_path /path/to/base-checkpoint-wrapper/hf_ckpt \
  --recap_adapter_path /path/to/recap_adapter.safetensors \
  --recap_condition positive
```

Artifact tensor names, shapes, SHA-256, signed-axis metadata, and base
verification counts are stored in the adjacent JSON sidecar.

Optional advantage-only CFG is enabled by choosing a scale other than `1.0`:

```bash
python -m deploy.lingbot_vla_v2_policy \
  --model_path /path/to/recap-checkpoint \
  --recap_condition positive \
  --recap_cfg_scale 1.5 \
  --use_compile false
```

For every Euler denoising step, the implementation duplicates the same state,
noisy action, and time into a `[positive, null]` batch and computes
`v_null + scale * (v_positive - v_null)` before updating the action. Each
condition receives its own prefix KV cache; positive cache entries are never
reused as null entries. Scale `0` selects the null branch, scale `1` selects the
positive branch without paying the doubled CFG compute cost, and scales above
`1` extrapolate toward positive-advantage behavior. CFG currently forces eager
inference because the fixed-shape doubled cache has not yet been benchmarked
under the nested compile path.

## Corrected `click_bell` result

The causal decision-1 dataset contained 32 balanced samples from six
mixed-outcome environment states (16 positive and 16 negative). The final
signed model trained a rank-8 velocity LoRA only on the 16 causal-positive
samples while all 1,708 official tensors were kept elementwise unchanged.
Evaluation used FP32, policy seed 42, random task
instructions, compiled inference (the historical launcher's default), and a
process-global policy RNG stream. It did **not** use the later per-episode
common-noise protocol. Three fresh environment-seed groups were not used for
collection or model selection:

| Condition | Seed group 24 | Seed group 25 | Seed group 26 | Aggregate |
| --- | ---: | ---: | ---: | ---: |
| positive, all decisions | 19/20 | 20/20 | 20/20 | **59/60 (98.33%)** |
| null / frozen base | 16/20 | 16/20 | 14/20 | **46/60 (76.67%)** |
| signed negative, all decisions | 13/20 | 12/20 | 10/20 | **35/60 (58.33%)** |
| positive, decision 1 only | 19/20 | 17/20 | 16/20 | 52/60 (86.67%) |

For paired positive versus null outcomes there were 14 improvements, one
regression, and 45 ties (two-sided exact McNemar/binomial p=0.0009765625).
For null versus negative there were 12 improvements, one regression, and 47
ties (p=0.00341796875). Seed group 26 loaded the compact artifact over the
untouched official base checkpoint. This resolves the diagnosed `click_bell`
instability, but it is not evidence for a 50-task improvement until the same
protocol is repeated across tasks and larger held-out sets.

## Replay regression audit (2026-09-08)

Do not equate an environment seed with a causal intervention state. The V2
holdout controller changed continuation seed 200 to 300 and intervention index
1 to the default 0. The old deterministic instruction generator also used
`test_num` as its candidate-list length: collecting 30 episodes and evaluating
3–5 changed the text for the same task/seed. In the initial three-task audit,
10/13 instructions changed and 0/13 intervention observations matched the
collected states within their declared tolerances. These rollouts remain
cross-protocol policy-generalization measurements, **not** fixed-state causal
replays and not evidence that action/outcome signal is absent.

Deterministic text now uses a fixed 32-description budget, isolates Python and
NumPy RNGs, and records `instruction_protocol=task-seed-v2-fixed-candidates32`.
This is a protocol version change. To replay an existing dataset, freeze its
**literal collected text**, rather than regenerate it with either implementation:

```bash
python scripts/recap_audit_causal_replay.py \
  --input /path/to/state_splits/click_bell/holdout \
  --policy-seed 900 --output /dev/shm/click_replay_protocol
```

The generated directory contains environment, instruction, and decision maps,
plus the collected continuation seed in `protocol.json`. Pass those maps with
`--recap_environment_seed_map`, `--recap_instruction_map`, and
`--counterfactual_policy_decision_map`, and preserve the continuation seed and
`--common_noise_per_episode`. Set `--test_num` to the task's exact seed count;
run tasks with different counts/schedules separately. Audit a null rollout by
adding `--null-rollouts /path/to/null` to the command (use a new output directory).
A mismatch exits nonzero; do not bypass the check by increasing tolerances.

Always include the previously successful artifact as a positive control before
changing the architecture or training objective. A 20-state diagnostic crossover
on historical group 26, with eager FP32 and modern common noise, obtained
null 16/20, v22-all 19/20, v22-decision-1 18/20, and V2-positive-only-all 18/20
without retraining. The subsequent same-seed window/sign ablation obtained V2
decision-1 16/20, V2 signed-negative-all 14/20, and v22 signed-negative-all 11/20.
Thus V2-all gave 90% > null 80% > signed-negative 70%, while restricting V2 to
decision 1 tied null. Correctly replaying the three original causal holdouts
restored observation matching (3/3) and gave V2-positive 2/3 vs null 1/3.
These small exploratory comparisons do not establish fresh statistical
significance or authorize promotion. Compare arbitrary variants with
`scripts/recap_compare_policy_variants.py`; it requires identical planned seeds,
text, and noise metadata and reports underpowered comparisons as inconclusive.

The signed adapter is `sign * B A h(state, noisy_action, time)`, **not** a fixed
global action vector. Opposite raw action differences across states do not by
themselves prove it cannot learn. Likewise a raw-action linear permutation probe
is a useful diagnostic, not a necessary condition for a state-conditioned model.
See [the investigation report](recap_root_cause_20260908.md) for evidence and
remaining limitations. Keep new/final seed groups untouched during diagnosis.

## Scaling safely to the 50-task suite

Do not apply one task's residual to every task. Train compact task-specific
adapters and deploy them through a checksum-verified registry:

```json
{
  "schema_version": 1,
  "tasks": {
    "click_bell": {
      "path": "adapters/click_bell.safetensors",
      "sha256": "...",
      "condition_start_decision": 1,
      "condition_decisions": 1
    }
  }
}
```

```bash
python -m deploy.lingbot_vla_v2_policy \
  --model_path /path/to/adapter-enabled-base-wrapper/hf_ckpt \
  --recap_adapter_registry /path/to/registry.json \
  --recap_condition positive
```

The RoboTwin client sends the canonical task name and environment seed at
reset. The server swaps only the tiny adapter tensors. Registry entries can
restrict the task adapter to a causal decision window with
`condition_start_decision` and `condition_decisions`. A task absent from the validated registry is
forced to null/base even if the server default is positive. This makes already
perfect tasks and failed validation candidates safe regression guards.

For collection across tasks, first record common-noise reference rollouts, then
run `scripts/recap_plan_task_interventions.py`. It derives the modal successful
terminal decision per task and emits a strict decision map. Pass it with
`--counterfactual_policy_decision_map`. For broad collection, use
`--recap_deterministic_instructions`: it chooses generated text by a stable
hash of canonical task and environment seed, including valid seeds that a
reference run may have skipped during environment initialization. A frozen
`--recap_instruction_map` remains available when it is known to cover every
valid seed. Thus every branch varies the planned action while reusing exactly
the same task instruction. Use
`scripts/recap_select_task_pairs.py --pairwise-match` to select balanced,
tolerance-valid mixed-outcome pairs per task. Tasks with no mixed group must
collect another candidate decision or continuation schedule; they must not
receive outcome-BC labels. `--common_noise_per_episode` keeps random numbers
paired across conditions while varying them by task/environment/decision. For
validation, build a frozen map with
`scripts/recap_build_environment_seed_map.py` and pass
`--recap_environment_seed_map`; fixed seeds are retried and failure is explicit
instead of silently substituting different seeds.

## Recommended experiment

1. Select difficult target tasks and easy regression-guard tasks.
2. Freeze train, validation, and untouched held-out seed groups in advance.
3. Identify a decision likely to affect the outcome; keep environment, prompt,
   prefix actions, and continuation noise fixed while varying only that action.
4. Keep only mixed-outcome observation groups, verify observation proximity,
   and balance outcomes within each group.
5. Train a null-preserving velocity adapter; do not update base tensors.
6. Use a signed axis instead of fitting an independent negative BC policy.
7. Compare positive, null, and negative with identical seeds and report paired
   discordances and confidence intervals.
8. For decisions that cannot be intervened on, fit a visual-language value
   model and reject uncertain labels rather than propagating episode outcomes.

The go/no-go criterion is `positive > null > negative` on fresh held-out seeds,
with null exactly preserving the base policy and no material regression on easy
guard tasks.

## Deterministic confirmatory evaluation

Common action-noise seeds alone do not make the custom inference MoE repeatable.
Use `--deterministic_algorithms True` on the policy server or RoboTwin launcher
for strict paired evaluation. The flag enables stable route packing and
fixed-order expert-output reduction; it defaults to False for legacy numerical
compatibility. The CLI supplies `CUBLAS_WORKSPACE_CONFIG=:4096:8` if unset.
Base-only adapter wrappers now initialize absent adapter tensors safely, and
nonfinite adapters/actions are rejected rather than counted as policy failures.
See the [Null reproducibility audit](recap_null_reproducibility_20260908.md).

The [50-task preregistration](recap_confirmatory_protocol.md) separates consumed-
seed controls, index-42 candidate selection, independent index-43–44 confirmation,
and a single frozen-registry final evaluation on indices 50–52. The claim is
an equal-weight mean improvement on the fixed suite, not improvement on every
task. An incomplete or nonsignificant study is not a successful demonstration.

`scripts/recap_confirmatory_study.py` snapshots code/weights, freezes cohorts and
decisions, audits actual server telemetry, and stages checksummed rollout evidence.
`scripts/recap_study_status.py STUDY_ROOT` reports progress without aggregating
partial final outcomes. See the [runbook](recap_confirmatory_runbook.md) for paths
and safe continuation commands.
