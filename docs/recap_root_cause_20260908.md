# RECAP 失效调查（2026-09-08）

## 结论边界

**没有证据证明 RECAP 机制整体失效。已在当前推理代码下重新观察到 80% → 90%/95%。**
此前把小样本、换协议的 holdout 未通过，进一步解释为“因果方向冲突、单一轴不成立”，证据不足。
已经定位并修复的是协议错误和缺失的诊断检查；不是已经证明 50 个任务泛化问题全部解决。

这次没有重新训练、没有修改任何 base/adapter 权重、没有推广新 adapter，也没有使用
fresh indices 42–44 或 final indices 50–52。

## 1. 交叉复验：旧 adapter 没有坏，新 V2 也并非完全无效

复用历史诊断组 26（环境 seeds 2700000–2700019），不是新的最终测试集。
首轮四个条件及追加的三个窗口/negative 条件使用相同的官方 base、相同 YAML、FP32/eager、policy seed 900、
continuation seed 300、counterfactual index 0、逐 episode CRN、相同指令与完整 seed manifest。
模型推理代码为 `f5aacee`，没有为了获得结果修改模型代码。

| 模型与启用窗口 | 成功 | 相对 Null 的改善 / 退化 | 双侧 exact p |
|---|---:|---:|---:|
| Null/base | 16/20（80%） | — | — |
| 历史 v22，全程 | 19/20（95%） | 3 / 0 | 0.25 |
| 历史 v22，仅 decision 1 | 18/20（90%） | 2 / 0 | 0.50 |
| V2 positive-only，全程 | 18/20（90%） | 2 / 0 | 0.50 |
| V2 positive-only，仅 decision 1 | 16/20（80%） | 1 / 1 | 1.00 |
| V2 signed-negative，全程（同一个 positive-only artifact 取反） | 14/20（70%） | 0 / 2 | 0.50 |
| 历史 v22 signed-negative，全程 | 11/20（55%） | 0 / 5 | 0.0625 |

因此同一个 V2 artifact，在同一组 seeds 上：**全程 90% > Null 80% > Negative 70%**；
只改成仅 decision 1，则为 80%，与 Null 持平。这个窗口消融不需要重训，也不需要换架构。
这证明“部署窗口变化”确实改变了本次测量，不能把相应差异全部归因于训练失败。

首轮每个 variant 的 20 个初始观测与 Null 均逐字节一致；指令及噪声元数据也一致。
全程 v22 的首 chunk 动作差异 RMS 均值约 0.00439，V2 约 0.000963，不是 adapter 没有加载。
仅 decision-1 的模型在首 chunk 为 Null，但不同 GPU 的底层计算仍有约 0.000050 RMS 漂移；
不能把数学上的零 residual 误写成跨 GPU 动作逐位一致。

**20 个 episode 不足以宣称新的显著提升；上述 p 均不显著。** 但这个正对照直接反驳了
“旧实现已经损坏”或“所有 V2 学到的东西都无效”的绝对判断。不能把 18/20 当成 50-task 的 90%。

历史 59/60 对 46/60 的实验与新版至少有这些差异：全程/单 decision、训练集、padding
loss、随机指令生成、compile/eager、全局/逐 episode 动作 RNG、评测状态和样本量。
原来的 inference YAML 与当前 wrapper YAML 已比较，一致；架构 YAML 不是这里的差异。

## 2. 确认的代码错误：deterministic 指令依赖 test_num

原调用：

```python
results = generate_episode_descriptions(task, episode_info, test_num)
instruction = candidates[hash(task, seed) % len(candidates)]
```

RoboTwin 第三个参数是 **description 数量**，不是场景随机种子。
采集 `test_num=30`，holdout 改为 3/5，候选集合长度、取模下标、unseen 生成前的 RNG
消耗均改变。仅固定 Python RNG seed 不能修复这个问题。

原始 V2 initial holdout 与其所留出的采集样本相比：

- click_bell：3/3 指令不同；
- place_can_basket：4/5 指令不同；
- stack_bowls_three：3/5 指令不同。

修复：`deploy/recap_instructions.py` 使用固定 32-description budget，保存/恢复 Python 和
NumPy RNG，并记录协议版本。legacy random 模式不改。已有数据必须使用从 manifest
导出的原始指令 map，不能用新生成器重建旧指令。

## 3. 确认的实验脚本错误：只保留 environment seed，没有保留 intervention state

采集 R2 使用 continuation **200**、counterfactual decision **1**。
旧 holdout controller 使用 continuation **300**，并遗漏 decision 参数，回到默认 **0**。
这意味着 decision 0 前缀就改成另一组噪声，加上指令变化，decision 1 时已是另一状态。

对三个任务的 13 个 holdout 做原始 NPZ 审计：

- continuation mismatch：13/13；
- counterfactual index mismatch：13/13；
- instruction mismatch：10/13；
- 在原阈值（state max-abs ≤ 0.001、image MAE ≤ 2.0）内匹配的 intervention state：**0/13**；
- 最接近采集状态的 joint max-abs 差异约 0.00225–0.07759。

这些 rollout 在各模型条件之间仍可作为**换协议后的完整策略泛化测量**，不能说分数是假数据；
但它们不是原先宣称的固定因果状态回放，也不能由此断言采集到的因果动作信号不存在。

新增 `scripts/recap_audit_causal_replay.py` 从 holdout manifests 导出字面指令、seed list、
intervention map 与 continuation schedule，拒绝混用不兼容 schedule，并检查 Null 回放观测。
使用这些导出的协议重放 click_bell 后，**3/3 状态重新通过原始 tolerance**：最大 nearest
joint drift 为 0.0000791，而不是之前的 0.0178。没有放宽阈值。
修复协议后的这三个状态为 **Positive 2/3、Null 1/3**（1 改善、0 退化，p=1.0）；
旧换协议测量为 Positive 2/3、Null 2/3，表明此前不是在测同一状态分布。
仍然不能用三个状态宣称显著泛化提升，但可以确认协议修复恢复了正确的因果回放。

## 4. 为什么不能接受上一份 handoff 的主要建模推断

- 实际 residual 是 `sign × B A h(state, noisy_action, flow_time)`，不是一个全局常数动作。
  它本来就依赖图像/语言/状态特征，可以在不同状态给出相反方向；有单元测试明确验证。
  原始 action delta 的跨状态 cosine 接近零，不等于此模型不能学习。
- 3–5 个 held-out states 太少。即使 5 个 paired discordances 全部改善，双侧 exact p
  也只能到 0.0625。Null=5/5 时，`Positive > Null` 在该样本上更是数学上不可能。
  “不推广”是合理安全决定，“已证明学习失效”不是。
- 所谓 signed paired 训练仍是带正负标签的普通 flow BC；当前 converter/trainer 不把
  pair/state ID 传进一个成对 ranking/contrastive loss。不能认为已充分测试真正的成对优化。
- 历史 v22 的 16 个 positive chunks 总计 800 timesteps，只有 **225 个真实执行**，
  **575 个（71.875%）是 terminal action 重复 padding**，旧 loss 全部监督。
  修复 padding 后训练目标发生了实质改变。因此历史 98% 不能作为“同一正确目标”应当
  恒定复现的指标。这个人工 hold-pose 目标可能产生有用偏置，但尚未做单变量重训消融，
  不能宣称它已被证明是提升的来源，也不能为了追回数字而重新打开错误的 padding loss。

## 5. 其他已修复的可复现性问题

多 GPU launcher 的 server 在 CRN 模式使用固定 policy seed，sim recorder 却仍记录
`policy_seed + slot`。这会产生错误的 rollout provenance。现在两处共用
`policy_seed_for_slot`。它不解释本次单 slot V2 的分数，但必须在下一轮多任务实验前修复。

新增 `scripts/recap_compare_policy_variants.py`：仅比较完整、同指令、同噪声计划的 seed 集，
去掉旧 retry、拒绝同 attempt 重复、保留 manifest SHA-256，并把小样本非显著差异标为
`inconclusive`，不自动推广任何 adapter。

## 6. 现在应该如何使用结果

1. 保留 v22 作为每轮都运行的正对照；必须带上它历史成功的全程窗口，不要悄悄改为仅 d1。
2. 因果 holdout replay 与跨噪声/指令的全策略 validation 分开记录，不能用后者替代前者诊断。
3. 先比较同一个 artifact 的窗口，再比较新旧训练；不要同时改所有变量后推断架构失效。
4. Null ceiling 或缺乏 discordances 的 holdout 应标为 inconclusive；增加预先规划的独立
   validation states，而不是挑选失败 seeds 或调宽 tolerance。
5. 目前不需要先实现 mixture/multi-axis 或把线性 raw-action permutation probe 变成硬 gate。
   真正的 paired/state-balanced objective 和 censored-chunk 输入问题可以作为后续受控实验。
6. 不修改生产 registry；获得足量新验证证据后再推广。缺失 task entry 会按设计强制 Null，
   因而“空 registry 下没有提升”不是加载失败。

## 原始证据与工具

远端权威诊断根目录：
`/dev/shm/recap_root_bridge_20260908`

持久化目录（报告、脚本、SHA-256、完整 rollout archive）：
`/vla-cd/ReconVLA/Lingbot-VLA/outputs/recap_root_cause_20260908`

- `plan.json`、`followup_plan.json`、`modern_full_comparison.json`、`initial_state_audit.json`
- `causal_replay_protocol/audit.json`：旧 13-state 测量审计
- `corrected_replay_audit/audit.json`、`corrected_replay_comparison.json`：修复后 click_bell 3-state 回放
- `instruction_generator_regression.json`：实际 RoboTwin generator 的 test_num=3/5/12/20/30 回归测试
- `rollouts/<variant>/attempt-1/click_bell/*/manifest.json` 及对应 NPZ
- `followup_plan.json`：V2 窗口和 negative 诊断，以及修复协议后的 positive 回放

历史 artifact：
`outputs/recap_click_bell_signed_causal_v22/adapter/recap_adapter.safetensors`

SHA-256：`b2904a1b9282a73eac958a0406e3a348f50884ff04becf29fd63966ebb6c78ba`

V2 positive-only artifact SHA-256：
`b195f6983a7dda5ffc8a2c66d994afcb7abdfb0a68a1d3907b4f9be7172e1580`

所有新诊断均先写 `/dev/shm`，不以可能损坏的 NFS `_result.json` 作为权威来源。
共完成 146 个诊断 episodes（7×20 + 2×3）；它们不是 146 个独立环境状态。
本地测试：**95 passed, 5 skipped**；新增工具 Ruff 检查、shell 语法检查通过。
