# LingBot Recap vs π0.6 / π*0.6

This document compares the Recap stack in this repository with Physical
Intelligence's π0.6 VLA and the RECAP method that produces π*0.6. It is a
map of what is aligned, rewritten, or missing. It does not change training or
deployment behavior.

Sources:

- π0.6 model card (Physical Intelligence, 17 Nov 2025)
- *π*0.6: a VLA That Learns From Experience* (arXiv:2511.14759)
- This repo: [`docs/recap.md`](recap.md), `lingbotvla/recap/`, `deploy/recap_*.py`,
  `tasks/vla/train_lingbotvla.py`

## Verdict

This repository implements an **engineered subset of RECAP** on **LingBot-VLA 2.0**.
It is not a π0.6 clone and it is not a full π*0.6 reproduction.

| Name | What it is | In this repo |
| --- | --- | --- |
| **π0.6** | PI's base VLA: Gemma3-4B + ~860M action expert, flow matching + FAST, Knowledge Insulation, hierarchical subtasks | **Not implemented.** The base model is LingBot-VLA 2.0 (Qwen3-VL-4B + MoE action expert + depth/video distillation) |
| **RECAP** | Train a VLA with advantage conditioning so RL becomes ordinary flow-matching / next-token loss | **Partial.** Labels, conditions, value primitives, causal collection, adapter training, and deployment exist |
| **π*0.6** | π0.6 after the full RECAP loop (offline RL pre-training + on-robot iterations + human interventions) | **Not reached.** No visual value encoder, no teleop corrections, no multi-round on-policy loop. Cross-task closed-loop gains are not established |

`tasks/vla/train_pi0.py` is a separate training entry. It does not implement Recap.

## 1. Base VLA

RECAP is a training protocol on top of a VLA. The two bases already differ, so
every later choice (where the condition is injected, which parameters move, what
the value model sees) diverges with them.

| | π0.6 | LingBot-VLA 2.0 |
| --- | --- | --- |
| VLM backbone | SigLIP ~400M + Gemma 3 4B | Qwen3-VL-4B-Instruct |
| Action expert | Same depth as the backbone, ~860M dense | Sparse MoE action expert |
| Action heads | Flow matching **and** FAST discrete tokens | Flow matching is the Recap path |
| Insulation / distillation | Knowledge Insulation: action-expert gradients do not enter the VLM; the VLM predicts FAST tokens and web co-training | Dual-query distillation from LingBot-Depth and DINO-Video, not the KI+FAST recipe |
| Hierarchy | Language subtask, then a continuous action chunk | Task text conditions the policy directly; no π0.5/0.6 high-level subtask head |
| Extra conditioning | Prompt metadata (style/quality); π*0.6 adds `Advantage: positive/negative` | Recap's recommended path is a velocity adapter, not metadata text |
| Target domain | Real robots: espresso, laundry, box assembly | RoboTwin 50-task sim, plus a real-robot config skeleton |

Matching the Recap *loss formula* does not make the models the same.

## 2. Original RECAP / π*0.6

Paper Algorithm 1 is a three-step loop:

1. Collect autonomous rollouts, optionally with expert teleop interventions.
2. Train a distributional value model on observations and language.
3. Binarize advantage into `I_t` and train the policy with the original
   supervised objectives, now conditioned on `I_t`.

Original mechanics:

- **Reward:** time-to-success. Intermediate steps get `-1`, success terminal `0`,
  failure terminal `-C_fail` (often 1000).
- **Value:** same VLA family, **smaller VLM backbone**, images + language, 201
  categorical bins, cross-entropy on discretized Monte-Carlo returns.
- **Advantage:** n-step (paper commonly uses N=50),
  `A = Σ r + V(o_{t+N}) − V(o_t)`, then a per-task quantile so roughly 30% of
  actions are labeled positive.
- **Condition injection:** text tokens after the high-level subtask and before
  action tokens: `Advantage: positive` or `Advantage: negative`.
- **Policy objective:** conditioned NLL on discrete tokens and flow-matching MSE
  on continuous actions. No PPO and no action log-probabilities.
- **Human interventions:** forced `I_t = True`.
- **Data mix:** demonstrations + on-policy successes/failures + intervention
  snippets in one loop.
- **Stages:** offline RL on large demonstration data produces π*0.6, then
  on-robot Recap specializes a task.
- **Inference:** positive condition, optional CFG between conditional and
  unconditional velocities.

The original system is **full-model text conditioning + a visual critic + a
real-robot heterogeneous loop**.

## 3. What this repo implements

Package: `lingbotvla/recap/`.
Deploy: `deploy/recap_*.py` and `deploy/lingbot_vla_v2_policy.py`.
Train: `tasks/vla/train_lingbotvla.py`.

### Aligned with the paper

| Original piece | Here | Where |
| --- | --- | --- |
| `Advantage: positive/negative` text | Yes; shared by train and serve | `conditioning.py` |
| positive / negative / null | Yes; null is unconditional | `RecapCondition` |
| time-to-success, MC return, `A = R − V`, binarization | Yes | `labels.py` |
| 201-bin categorical value + CE | Yes (`value_min=-2000`, `value_max=0`) | `value.py` |
| Condition dropout onto the null branch | Yes | `maybe_drop_recap_condition` |
| Serve positive by default | Yes | `lingbot_vla_v2_policy.py` |
| CFG `v_null + s (v_pos − v_null)` | Yes | `cfg.py` |
| Unlabeled expert demos treated as positive | Yes (`recap_missing_condition`) | `docs/recap.md` |
| Keep flow matching; no policy gradient | Yes | `loss.py` + adapter-only FM |

### Intentional rewrites

Episode success is **not** an action-advantage label. Successful and failed
episodes also differed in environment state, task wording, action noise, and
every later policy action. Labeling every decision with the final outcome
trained two behavior-cloned policies on a confounded split. That is why this
stack replaced observational advantage with a **single-decision intervention**.

| Original | Recommended path here |
| --- | --- |
| Advantage text into the VLM; finetune the policy | **Disable the prompt.** Freeze every official tensor and train only a **velocity LoRA** |
| Negative-conditioned BC | **Signed axis:** learn a residual from causal-positive samples only; `neg = −pos`, `null = 0` |
| Value labels every step, mixed signs inside one trajectory | Hold env seed, instruction, prefix, and continuation noise fixed; change **one** policy decision; keep only mixed-outcome groups |
| Real-robot teleop as forced-positive | RoboTwin counterfactuals via policy / continuation seeds; no teleop channel |
| One generalist π*0.6 | Per-task compact adapter + SHA-256 registry; unknown tasks are forced to null/base |
| Offline RL pre-training of the generalist | None. Recap is a post-training plugin |

Velocity residual (`lingbotvla/recap/adapter.py`):

```
r = (scale / rank) * B A h(state, noisy_action, time)
v_out = v_base + sign(condition) * r
```

- `null` residual is exactly zero (a zero-gradient graph edge remains so
  adapter-only training still works).
- `recap_signed_velocity_axis=true` reuses one A/B pair with a sign flip
  instead of fitting a second failure BC policy.
- This is a velocity identity at the **same hidden state**. Integrated actions
  need not be exact opposites.

`lingbotvla/recap/training.py` adds a fail-closed **deployment-feature**
backend: eager, FP32, fused MoE, real prefix KV cache, and only
`recap_velocity_lora_a/b` trainable. Training `h` and `v_base` must match
serving, not merely share YAML names.

### Documented gaps

[`docs/recap.md`](recap.md) already lists work that is required before a
task-suite-wide RECAP claim:

1. A **trainable smaller VLM** critic. `VLMPooledValueModel` now attaches
   `CategoricalValueHead` to frozen LingBot prefix embeddings; that is not the
   separately trained 670M-scale VLM from π*0.6.
2. Repeated counterfactual collection/training rounds across tasks.
3. Held-out evaluation on all target tasks, not only the `click_bell`
   diagnosis.

Shipped toward the paper, still not the paper:

- n-step advantages (`--n-step 50 --n-step-unit actions` accumulates executed
  action length; `decisions` still counts chunks) and per-task `--positive-fraction`
- Forced-positive labels for `"intervention": true` steps (no teleop collector yet)
- Camera CNN critic (`--encoder visual_task`)
- Frozen VLM-pooled critic (`--encoder vlm_pooled` + embedding cache)

Still missing relative to π*0.6:

- A smaller VLM value model in the same family as the policy
- Human teleop collection / DAgger-style corrections on the robot
- Advantage-conditioned **offline RL pre-training** on large demonstration data
- Hierarchical subtask tokens with the advantage text inserted after them
- FAST / Knowledge Insulation dual heads

`StateTaskValueModel` is a proprioception + task-id MLP. `VisualTaskValueModel`
adds cameras. `VLMPooledValueModel` is the frozen-VLM head. None of them is
π*0.6's separately trained smaller VLM critic.

## 4. Data loop

```
Original RECAP
  D_demo --> V_pre --> π_pre (offline RL, text Advantage)
                |
                v
  on-robot rollout ± human intervention
                |
                v
  update D --> retrain V --> retrain π --> repeat

This repo
  frozen LingBot-VLA 2.0
                |
                v
  RoboTwin causal paired collection (common noise, one-decision intervention)
                |
                +--> mixed success/fail pairs --> causal-positive --> signed velocity LoRA
                +--> optional state+task value labels (visual value not shipped)
                |
                v
  per-task adapter registry; default positive; optional CFG
```

Collection is stricter than the paper in simulation, and narrower in domain:

- `deploy/recap_rollout_recorder.py` stores one NPZ per decision, including the
  generated chunk and the executed prefix.
- `lingbotvla/recap/noise.py` pairs continuation seeds and per-episode common
  random numbers.
- `scripts/recap_select_paired_decisions.py` matches observations and balances
  mixed-outcome groups.
- Locked cohorts, literal instructions, initial-state hashes, and
  `--deterministic_algorithms` keep paired eval from silently changing the
  state.

Those controls repair simulator confounding. They are not the paper's real-robot
intervention protocol.

## 5. Empirical status

Having the modules is not the same as reproducing the method.

| Experiment | Result | What it supports |
| --- | --- | --- |
| `click_bell` causal decision-1 | positive 59/60 vs null 46/60 vs negative 35/60; McNemar p ≈ 0.001 | The signed adapter works **on that task under that protocol** |
| 50-task confirmatory (2026-09-10) | 6000 paired episodes; macro 92.87% → 93.13% (+0.27 pp); **only `click_bell` loaded an adapter** | Pre-registered suite-mean claim, not Recap on every task |
| P0 repaired-policy check (2026-09-11) | hanging_mug / place_can_basket / stack_bowls_three: macro **−1.67 pp**, CI includes 0 | Training/serving feature parity and full-rank init did **not** become closed-loop gain |

Implementation status: the Recap toolchain can collect, train, serve, and audit.
Scientific status: the original “visual critic + text-conditioned full model +
real-robot loop” is not reproduced. The causal-adapter variant holds on
`click_bell` and has not been shown across tasks.

## 6. Side-by-side

| Layer | π*0.6 RECAP | LingBot Recap | Match |
| --- | --- | --- | --- |
| Base VLA | Gemma3-4B + dense 860M expert | Qwen3-VL-4B + MoE expert + depth/video | Different architecture |
| Condition channel | Text tokens into the VLM | Velocity LoRA by default; text exists but is not recommended | Rewrite |
| Trainable set | Advantage-conditioned policy SFT/FM | Only LoRA `A/B`; base tensors frozen elementwise | Rewrite |
| Negative condition | Separate `Advantage: negative` BC | Sign flip of the positive residual | Rewrite |
| Value | Smaller VLM, images + language, 201 bins | 201 bins; state+task MLP, camera CNN, or **frozen VLM pooled head** | Head aligned; encoder frozen, not a trained smaller VLM |
| Labels | V(s) advantage ± forced-positive interventions | Single-decision counterfactual success/fail | Stricter, narrower |
| Data | Demos + on-policy + teleop | RoboTwin paired rollouts | Different domain |
| Pre-training stage | Offline RL → π*0.6 | None | Missing |
| CFG | Conditional vs unconditional velocity | Positive vs null velocity | Aligned |
| Evaluation | Real-robot throughput / failure rate | Pre-registered paired McNemar + Holm + locked cohorts | Lab protocol |

## 7. What not to claim

- Do not call this repo a π0.6 implementation.
- Do not call a Recap-enabled LingBot checkpoint π*0.6.
- Do not turn `recap_prompt_enabled` back on just to “look more official”. The
  base checkpoint never trained on an `Advantage:` prefix; a zero-update
  positive prompt already changes the policy.
- Do not scale the 50-task training loop from backend/init repairs alone. The
  2026-09-11 three-task check already rejected that expansion.

Closer alignment with original RECAP is a separate, frozen research plan:
visual value on VLM pooled observations, n-step advantage labels, a controlled
comparison of adapter vs text-conditioned expert updates, and multi-round
collect → relabel → retrain on **one** task with a stated failure-stage
hypothesis.
