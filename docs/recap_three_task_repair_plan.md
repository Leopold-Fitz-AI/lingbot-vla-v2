# 三任务 RECAP 修复计划

依据：[2026-09-09 训练链路审计](recap_three_task_training_diagnosis_20260909.md)。
本文件是**下一研究版本的实施方案，不是对当前确证实验的修改或启动许可**。
以下新实验预算/门槛需在分配不重叠的 seed IDs、确认资源后整版冻结，才能执行。

## 0. 不动的边界

- 当前 `_locked_cohorts` 实验继续使用已冻结的代码、registry、6000 primary + 60 Negative
  最终预算及原分析；其失败/阴性结果也必须完整报告。
- 不把正在运行的 indices 50–52 用于误差分析、挑模型、调 scale、训练或选窗口。
  已经看过的 index 42 和 43–44 不再是新方案的独立验证数据。
- 原 base、13 个 adapter、全部旧失败实验保留。不覆写任何 frozen source、launch marker
  或决定文件；不把新结果替换成旧实验“恢复成功”。
- 新版本仍只允许 GPUs 4–7；当前评估占用期间只做 CPU 审计/单元测试。
- 任何 policy evaluation 都不得按成功率重试、丢任务、换种子、放宽初态容差。
  专家预检和同 cohort 基础设施重试沿用明确上限；耗尽即停止。

## 1. P0：先统一训练与部署的冻结函数

### 并行的现有轨迹诊断（不需要新 GPU rollout）

对已完成确认的全部 **70 个 discordant pairs**（10 hanging、31 basket、29 bowls），
加每任务按 seed 固定取 5 个 shared-success 和 5 个 shared-failure pairs 作控制。
导出 condition-blind storyboard、joint/gripper/action 时间线，标注最早观测分叉、抓取、
携带、放置/叠放、超时和无法判断。不能从最终 success bool 猜失败阶段，也不能从
相机图像伪造没有记录的接触力/物体速度。输出 `failure_stage_audit.json` 和原始 NPZ hash。
它决定后续 anchor/retention 数据应覆盖哪里；仍属探索诊断，不产生新的显著性声明。

### 改动目标

文件：
- `lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py`
- `lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py`
- `tasks/vla/train_lingbotvla.py`
- 新增一个 frozen-feature cache/training 工具（独立于现有实验）。

为 `train_recap_adapter_only + velocity_lora + recap_prompt_enabled=False`
增加显式、可审计的训练路径：

1. 复用部署端的 prefix/denoise feature 函数，在 `.eval()`、`no_grad()` 下计算
   相同 FP32 tensors / deterministic robby backend 的 `h` 和 `v_base`。
2. 离开 `no_grad()` 后再计算可训练的 `BAh` 和 loss，使 A/B 正常收到梯度。
3. 训练和推理调用同一 frozen feature 接口，记录实际 backend、dtype、路由、kernel
   版本和 TF32/Triton dot 设置，不能仅凭配置中的 `enable_fp32` 命名。
4. 对 embedding adapter、可训练 backbone、可训练 prompt、端到端 ODE 反传等不满足
   条件的模式拒绝此捷径。固定 teacher-forced x_t 的 adapter-only FM 可安全 detach；
   如果 x_t 本身依赖要训练的参数，不能随意切断该梯度。
5. 可以缓存 `(state_id, pair_id, action_id, continuation_id, noise_seed, t, h, v_base)`，
   让后续小 adapter 消融不再反复运行 6B backbone。缓存必须带 base/config/kernel SHA。

### 固定观测诊断，不跑新模拟器

使用原训练+旧 holdout 的全部 **50 个环境状态**（13 hanging + 20 basket + 17 bowls），
每个状态按固定顺序选一个已保存观测。固定 3 个 t 和 4 个 noise draws：
共 600 个输入，比较原训练路径、严格部署路径、新训练路径。

输出 `backend_parity.json`：router top-k 一致率、hidden/velocity max/RMS、同 adapter
残差幅度和路径误差比值。不能把“权重 checksum 一样”当作这个测试已通过。

### 验收

- 同设备、同 shape、同输入的“新训练 frozen features”和部署接口字节一致；若不能，
  先定位 cache/attention/kernel 分叉，不进入大规模重训。
- 初始 B=0，Null/Positive/Negative 都与 base 行为一致；更新后 Null residual 仍精确零。
- 第一更新步 B 有梯度，后续 A/B 有梯度；所有 base 参数无梯度且导出后 1708 tensor 不变。
- 非有限值、后端/精度 fallback 一律报错；不能为通过测试偷偷改成 BF16。
- 不因这个训练修复更改当前已测量的推理数值协议。

## 2. P0：修正退化初始化，但单独测其贡献

文件：`FlowMatchingV2.reset_recap_adapter` 与 `tests/test_recap_initialization.py`。

新增 versioned 初始化类型：局部 RNG 生成的满行秩正交 A，B=0。
A 的 Frobenius 范数与旧 sin 初始化匹配，避免把增大幅度混进秩消融。
局部 generator 不改变全局 Python/NumPy/Torch RNG；device/dtype/seed 写入 sidecar。
旧初始化作为兼容选项保留，既有 artifact 正常覆盖初始化。

验收：rank=8 时 8 个奇异值非退化；B=0 和 Null 精确保持；RNG 保存；加载旧权重
逐元素一致。不能只检查所有值 finite，也不能把 nominal rank 改大当作修复完成。

### 2×2 最小归因实验

| 因素 | 旧 | 新 |
|---|---|---|
| 冻结特征数值路径 | 原训练 fused/BF16 | 部署一致的 deterministic path |
| A 初始化 | sin、近 rank-2 | 同范数满秩正交 |

3 个任务 × 4 配置 × 3 个预声明 optimizer seeds = **36 次小 adapter 训练**。
此阶段保持每个任务原来的 objective、数据、50 steps、batch、lr，不同时加数据或改 loss。
比较训练/旧 holdout 的 paired preference surrogate、FM loss、残差谱、跨 noise 稳定性。
旧 holdout 只用于诊断，不计算“新的独立显著提升”。
不按最好的 optimizer seed 选正式模型；正式部署 seed 在任何新验证前固定，其他 seeds
只检查稳定性。若改善只存在训练 loss 或单一 seed，不能宣称解决了泛化。

## 3. P1：重建多阶段、可复验的训练信号

### 首轮有上限的 pilot

每任务先冻结 16 个 expert-feasible 训练环境状态，保留所有状态，包括 all-success /
all-failure 的情况。不得只挑 base 失败状态来表示一般部署分布。

- hanging：接近/抓取、对准挂放两类 anchor。
- basket：抓取、送入/释放两类 anchor。
- bowls：初始抓取、中间叠放、后续叠放三类 anchor。
- 每个到达的 anchor 4 个候选 action chunks，每个 action 使用同一组 4 个 continuation
  noise schedules。比较 action 时共享 continuation draws，而不只看单次成功/失败。
- 48 个参考 rollout，加每状态两个额外相同条件 A/A rollout（96）；
  分支上限 `16 * (2+2+3) * 4 * 4 = 1792`，总上限 **1936 个 policy rollouts**。
  重复 continuation 不算新的独立环境状态。
- 未到达的 anchor 明确记为 `not_reached`，不伪造样本或用另一个更好 seed 填空。
  这是训练 pilot 上限，不是允许缩短确认/最终评估 cohort。

### 必须改的 schema/collector

文件：`recap_select_paired_decisions.py`、`recap_rollouts_to_lerobot.py`、
`lingbotvla/data/vla_data/utils.py` 以及新 counterfactual collection driver。

保留并传进 learner：environment/state/pair/action/continuation IDs、原 decision、phase、
字面 instruction、prefix noise/array hashes、generated chunk、executed mask、Q/advantage
估计与不确定性、原始 manifests SHA。state/phase 分组采样，不能把同状态多个分支
当成多个独立训练状态。

比较前验证：同 literal instruction、同完整 prefix actions、同 initial/intervention
observations、可取得的 robot qpos/qvel、物体 pose/velocity、控制器状态。
A/A 必须没有 outcome discordance；若有则停止该采集版本，不能用 tolerance 掩盖。
历史容差仍不放宽，新的精确审计是附加约束。

4 个 continuation 只足够做 pilot 的噪声敏感性/半组符号一致性检查，**不是每个 action
都能获得显著 Q 排序**。报告置信区间与 ties，弱证据 abstain，不强行二分。
如果 pilot 显示符号高度不稳定，停止训练目标搜索；先冻结新的 continuation/anchor
诊断版本，而不是拿不可靠 label 加大模型。扩大数据的预算要在新的验证前另行冻结。

不把 simulation-only object states 偷偷作为部署观测。它们可用于审计/训练阶段标注；
部署 gate 只能依赖实际可获得的 RGB、proprioception、task。

## 4. P1：从“signed BC”走向可检验的偏好目标与保持能力

先恢复 pair/state schema 与 sampler，再实现 loss；不能先写一个叫 pairwise 的 loss
却继续随机拆散正负样本。所有模型固定使用 P0 的相同数值路径、初始化和同一份数据。

预声明 5 个 objective/部署候选：

1. positive-only FM BC（保留有效的简单控制）；
2. signed outcome FM BC（现有机制控制）；
3. positive FM + within-state preference surrogate；
4. 3 + 原本可靠状态上的 frozen-teacher retention；
5. 4 + observation-conditioned phase/confidence gate。

同一 pair 共享 t/noise draws，按有效动作维度与执行 mask 归一化。
可用 `S_theta(a|s)=-(FM_error_theta-FM_error_base)` 构造 paired logistic margin；
这是 **flow-error preference surrogate，不是精确 log-likelihood/DPO**。需要先验证
它与控制分支的成功排序相关，并保留 BC 正样本项，不能允许单纯把坏动作的误差做大。
弱 Q 差异降低权重/abstain，不把所有 0/-1000 元数据当成同等可靠的 advantage。

Retention 使用 baseline 自然 rollout/teacher latent 路径上的同输入 velocity 保持，
不只使用 mixed states 的 residual L2；这是保持能力 surrogate，不是精确动作 KL。
闭环中仍必须真实执行 guards。phase gate 使用冻结观测特征，而非硬把所有环境的
第二个 chunk 当成同一操作阶段；不直接跳到大 mixture/multi-axis 架构。

3 任务 × 5 候选 × 3 optimizer seeds = **45 次小训练**。
正式候选每配置只使用预先指定的那个 seed；其余用于稳定性诊断，不能 online 挑最好 seed。
样本增多后的 update 数按冻结的数据遍历规则确定，而不是沿用 50 steps 后就说训练充分，
也不以 validation loss 到最低为无限续训依据。LR/正则/margin/训练轮数需在新验证前
一次性写入 candidate manifest；不在当前 43–44 或最终 50–52 上校准。

## 5. P2：单独证明全策略泛化，不能把 replay 当成功率验证

新的 seed IDs 必须先查全局使用台账；本计划不臆定任何具体编号尚未使用。
训练、诊断、选择、确认、最终按环境 state 分割，同 state 的所有分支必须同 split。

建议冻结预算（独立于当前实验，实施前审定）：

| 阶段 | 设计 | Policy episodes |
|---|---|---:|
| 新模型选择 | 3 tasks × 80 states ×（5 candidates + shared Null） | 1440 |
| click_bell 正对照 | 80 states ×（v22_all + Null） | 160 |
| 独立确认 | 4 tasks × 200 states ×（Positive/Null/Negative） | 2400 |
| 新最终主评估 | 固定 50 tasks × 60 paired states × 2 policies | 6000 |
| 新最终 Negative | 仅准入 tasks | ≤240 |

上述 pilot 与后续验证的总预算上限为：
`1936 + 1440 + 160 + 2400 + 6000 + 240 = 12176`，不含单独冻结的训练数据扩充、
expert-only preflight 和明确有界的基础设施开销。P0 的 81 次训练/固定观测 forward
不是这些模拟器 policy episodes。预算较大，因此 P0/P1 未通过前不启动整套新验证。

选择规则仍提前固定：最大 paired net gain、最少 regressions、固定 priority。
新确认沿用四任务 Holm、Positive > Null > Negative、至少 +5 pp 实用收益，不能从
Negative 的描述性高分反选方向。若无候选通过，接受 `no_validated_candidate`。
最终仍完整测量 matched Null 与冻结路由政策；相同的 macro/CI/exact-test/guards 原则，
不合并旧最终结果凑显著，不补样本直到 p<0.05，也不自动改生产 registry。

## 6. 必交付的测试与报告

- `backend_parity.json`：实际 train/eval dtype/backend/hidden/velocity 对齐，梯度检查。
- `initialization_ablation.json`：同范数、秩、RNG、Null、旧 artifact 兼容性。
- `causal_collection_audit.json`：所有 pair/prefix/A/A/continuation 证据，rejections 与未到达
  anchor 透明记录；对 label 不确定性不作伪精确解释。
- `paired_objective_tests`：同 pair 同 noise/t、state-disjoint splits、no singleton/duplicate
  pair、padding 不监督、无未来 outcome 泄漏、无非 RECAP 参数更新。
- `candidate_manifest.json`：数据/代码/权重/采样/optimizer seeds 与损失/候选优先级冻结。
- `selection.json`、`confirmation.json`、`final_result.json`：严格分阶段；全任务贡献、
  回归例数、统计不确定性。阴性也交付。

## 实施次序

1. **现在已完成**：只读代码/训练数据/已完成确认审计及本计划；没有新训练/模拟器调用。
2. 当前最终实验结束或 GPUs 4–7 获准空闲后，另立版本执行 P0，先判断数值桥接与秩因素。
3. P0 通过才做小规模多阶段多续跑 pilot；信号不稳定则回到 collector，而不是堆 rank/LR。
4. schema、数据和目标通过离线/固定状态检查后才冻结新候选，进入独立全策略验证。
5. 无论新研究如何，都保留并报告当前研究的原始结论。
