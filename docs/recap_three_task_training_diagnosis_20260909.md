# 三任务 RECAP 无净收益：训练链路审计（2026-09-09）

## 结论与边界

这次审计没有训练模型、没有调用模拟器/策略、没有修改任何权重或正在运行的
确证实验，也没有读取 `evaluations/final`、`final_pairs.json` 或最终成功率。
只检查 V2 原始训练数据、代码、artifact，以及已经完成的筛选/独立确认数据。
CPU 审计不占用当前最终实验的 GPU；后续 GPU 实验必须等 GPUs 4–7 空闲。

**已定位具体实现问题和训练信号缺口，但尚未通过单因素重训确定各因素对成功率的
因果贡献。不能把所有退化归到一个 bug，更不能宣称已经修好这三个任务。**
最重要的新增发现是：训练/部署的冻结 MoE 不是同一计算路径；rank-8 初始化几乎
只有两个有效方向；训练实际只有 9/15/12 个独立环境状态、全部只覆盖 decision 1，
且所谓 advantage/pair 训练没有使用优势估计、置信度或成对目标。

详细执行方案见 [修复计划](recap_three_task_repair_plan.md)。

## 1. 先排除“未加载”“天花板”“当前配对错误”

| 任务/实际候选 | Positive/Null | 改善/退化 | 首个激活 chunk 动作 RMS，均值 |
|---|---:|---:|---:|
| hanging_mug / signed_all | 47/47 | 5/5 | 0.00009876 |
| place_can_basket / regularized_d1 | 71/70 | 16/15 | 0.00006702 |
| stack_bowls_three / positive_d1 | 80/79 | 15/14 | 0.00067880 |
| click_bell / v22_all（参照） | 100/84 | 16/0 | 0.00422726 |

每项都是 100 个独立确认状态；RMS 是原始 14 维动作 chunk 的描述量，
不同维度/任务的物理含义不同，**不能直接按这个比值放大 adapter**。
这里是实际 metadata 验证的 `demo_clean`，84% 是配套 Null，不是官方公布成绩。
尚未核实官方 checkpoint/recipe，不能声称已解释它与用户记忆中 >90% clean 分数的差距。

- 三个候选在 100/100 状态的激活动作均不等于 Null，且此前运行时 artifact/window
  校验通过，不是“加载失败所以没效果”。
- 三项在干预开始的观测均 100/100 字节一致。对 basket/bowls，激活前完整
  decision-0 数组在 Positive/Null/Negative 三个条件间也是 100/100 字节一致。
  hanging 从 decision 0 激活，不存在可检查的先行 policy chunk。
- 它们仍有很大提升空间，不是 Null 已 100%。失败在于收益与损害抵消。
- Negative 对 Null：hanging 6 改善/7 退化，basket 13/11，bowls 19/10。
  bowls 的 88% 对 79% 只是描述性优势，双侧 paired p=0.1360，**未证明反向显著更好**。
  这些机制诊断不是事后换方向的许可。

证据：`confirmation_action_audit.json`；仅读取已完成的 `evaluations/confirm`，
并记录了 3000 个 manifest/NPZ 的 SHA-256。

## 2. 数据覆盖：只有一个早期决策，且回报很远

从真正供训练的 `train.txt`、LeRobot conversion metadata、原始 NPZ 和
`state_splits/*/train` 核对，而不是拿采集总 episode 数当训练状态数：

| 项目 | hanging_mug | place_can_basket | stack_bowls_three | V2 click_bell 参照 |
|---|---:|---:|---:|---:|
| 独立训练环境 states | **9** | **15** | **12** | 9 |
| 正样本 chunks | **13** | **20** | **18** | 18 |
| signed 正/负 chunks | 13/13 | 20/20 | 18/18 | 18/18 |
| 原始干预 decision | 1 | 1 | 1 | 1 |
| 正样本干预后还需 policy decisions | **5** | **2–4** | **6–8** | 0–1 |
| 正样本真实执行长度 | 全部 50 | 全部 50 | 全部 50 | 均值 17.89，最短 7 |

所有 V2 配置都是 50 optimizer steps、global batch=4、micro batch=1、lr=0.001。
这意味着约 200 个训练样本呈现，不是 200 个新状态，也不是配置中 `num_train_epochs`
很大就真的执行了那么多 epoch。signed/regularized 共用同一份正负数据。

直接含义：

1. 没有 late-placement / second-or-third-stack 的训练动作信号。全程使用是在
   未覆盖状态上外推；仅 d1 使用又无法直接修复后续阶段。
2. 成功标签是一次特定未来噪声下的最终结果。hanging 的训练 continuation 混合
   200/400，另外两项为 200；部署为 300。每个比较 pair 没有跨多个 continuation
   的 Q/advantage 均值或不确定性估计。
3. 只有 mixed-outcome 状态被选入；没有为“原本稳定成功、不该改”的状态提供
   Positive 条件下的保持约束。Null 结构不变不等于 Positive 不会伤害这些状态。
4. click_bell 的回报多数就在干预 chunk 内；另外三项存在更长的信用分配链。
   这是数据事实，不是已经完成“长链必然导致失败”的实验。

**排除一条错误修复方向：这三个任务的正负训练 chunk 都执行满 50 步，没有
terminal padding；因此不能把它们这次不提升归因于 padding，也不能恢复错误的
padding loss 来追分。** v22/click 的 padding 是另一个需要独立研究的问题。

## 3. 确认的实现不一致：冻结权重不等于冻结计算函数

源码路径：

- `tasks/vla/train_lingbotvla.py`：`model.train()`。
- `lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py`：只有 CUDA +
  `not self.training` + `not torch.is_grad_enabled()` 才使用 `robby_moe_forward`。
- `lingbotvla/ops/fused_moe.py::fused_moe_forward`：其他 fused 路径把 hidden、
  routing weights、gate/up/down weights **显式转换为 BF16**。
- 当前严格推理使用 deterministic robby 路径，保留 FP32 tensor dtype，固定 pack/reduce。

这些任务的配置都是 `moe_implementation: fused`、`enable_fp32: true`、
`train_recap_adapter_only: true`。FSDP2 的 FP32 参数设置不能阻止内部显式 BF16 cast。
这里不是把 `enable_mixed_precision: true` 误当成所有网络都在 BF16：训练循环里所见
的 autocast 主要是关闭了的视觉教师分支；**关键是 MoE 算子本身强制转换**。
`f5aacee` 到确定性修复的 diff 也显示训练分支未被这一轮修复改掉。

于是实际目标为：

```
训练：v_train(obs, x_t, t) + sign(label) * B A h_train(obs, x_t, t)
部署：v_eval (obs, x_t, t) + sign(label) * B A h_eval (obs, x_t, t)
```

官方 1708 个 base tensor 逐元素未变，仍不意味着上述两个函数相同。
低幅度 residual 可能学习训练后端偏差，而不是稳定动作改善。
**路径/精度不一致已经证实；它造成多大 hidden/velocity 误差、能否解释成功率，
尚需冻结观测的同输入跨路径 replay 和单因素重训。** 此问题也存在于旧 v22 配置，
所以不能独自解释“click 有效而其他无效”。

FP32 在这里仍指 tensor dtype，不意味着把现有 Triton GEMM 换成全 IEEE-FP32。

## 4. objective 不是学到优势大小的 paired improvement

实际标签文件：

- 所有成功 chunk：`label=1, advantage=0, return=0, value=0`；
- 所有失败 chunk：`label=0, advantage=-1000, return=-1000, value=0`。

这是人工 outcome 标签，不是该动作的经过估计/校准的正优势。
`recap_rollouts_to_lerobot.py::_features` 不携带 pair/state/continuation IDs；
原始关联还保留在 source manifests/sidecars，可恢复，但没有进入 learner。
`FeatureTransform.feature_to_keep` 只保留 `recap_label`，不保留优势/置信度；
模型 forward 没有 pair grouping 或 advantage weighting。
`StateTaskValueModel` / categorical value primitives 存在，但不在这些 artifact 的训练链路里。

实际损失是：

```
x_t = t * noise + (1-t) * action
u_t = noise - action
L = masked_mean(|u_t - (v_train + sign(label) * B A h_train)|)
    + lambda * masked_mean((B A h_train)^2)
```

- positive：只用成功样本，lambda=0；
- signed：正负各半，仍是独立样本 L1 flow BC；
- regularized：**signed + lambda=100**，不是正样本 BC 加保持能力约束。

正负 residual 在相同 hidden 上取反，是结构事实，不是 `Positive > Null > Negative`
的数学保证；ODE 轨迹会改变 hidden，最终动作也不是严格互为镜像。
更不能误称 residual 是与 state 无关的全局常量。

这不证明“条件 BC 原理不成立”，或“RECAP 必須使用 DPO”。它说明现有实验没有
测试到真正的成对偏好/可靠优势目标。应保留 positive-only 作为控制，测试新增目标
是否比相同数据/数值路径下的 BC 更好。

lambda=100 也没有经过行为尺度校准：单个独立坐标的理想化
`|e-delta| + 100*delta^2` 最优解为把 e 截到 ±0.005；真实共享低秩模型不受这个
逐坐标硬界保证，但有明显缩小修正的倾向。这不是冻结基线成功状态的 trust region。
无正则候选也没稳定改善，所以不能把全部问题归咎于正则过强。
当前 micro batch=1，不应错误宣称短 chunk 在同 batch 中被长失败 chunk 按长度压倒。

## 5. 新发现：rank-8 初始化几乎是 rank-2

`FlowMatchingV2.reset_recap_adapter` 使用：

```
A[r,j] = 0.02 * sin((offset + r*768 + j) * 0.017)
B = 0
```

由 `sin(a+b)=sin(a)cos(b)+cos(a)sin(b)`，精确算术下所有行只张成两个方向。
实际 FP32 的初始 A 奇异值约：

```
0.8645, 0.6978, 1.93e-6, 1.61e-6, 1.18e-6, 1.05e-6, 8.78e-7, 7.66e-7
```

训练 A 可以离开这个子空间，不是永远 rank=2。对实际入选 artifact 的 BA，前两项
奇异值的平方能量占比为 hanging **99.793%**、basket **99.745%**、bowls **99.884%**。
这说明名义 rank=8 不等于已经学到八个有效方向。用实际 reset 方法的 CPU 测试已复现。

但谱集中本身不是过拟合/性能差的充分证据，v22 也使用相同初始化。
应做同 Frobenius 范数、局部 RNG、满行秩正交初始化的单变量消融；不要直接把 rank
加到 32，也不要改写已训练 artifact 或正在运行的模型初始化。

## 6. 旧“因果”数据的前缀并非完全一致

三个任务训练共有 **51 对**（13+20+18），原始 decision-0 generated actions
字节一致 **0/51**；干预前观测字节一致也 **0/51**。
前缀 action RMS 均值分别约 4.30e-5、4.44e-5、4.16e-5。
这些差异发生在选定的 decision 1 **之前**，不应解释为目标干预本身造成。

所有观察漂移仍在原先公开的容差内：各任务最大 joint drift 约
0.0001103/0.0001757/0.0001172，最大 image MAE 1.0548/1.3543/0.5308。
因此不能倒称当时的筛选违反了其容差；也不能把所有标签断言为错误。
已有旧 A/A 失配证明微小数值扰动可能改变结果，但这 51 对究竟多少受污染未知。
新收集必须使用已验证的确定性路径、逐 pair 前缀/观测审计及 A/A 假 discordance
检查；不能只修改评估端而把旧数据自动当成严格因果真值。

## 7. 证据优先级

| 优先级 | 已知问题 | 尚未证明的部分 |
|---|---|---|
| P0 | train/eval MoE 路径与计算 dtype 不同 | hidden/velocity 误差及其性能贡献 |
| P0 | 名义 rank-8 初始化只有两个主方向 | 满秩初始化是否提高泛化 |
| P1 | 9/15/12 states，全部 d1，回报延迟长；单 continuation outcome | 多阶段/多 continuation 数据能提高多少 |
| P1 | 没有优势/置信度或成对训练；regularized 只是 signed+L2 | 新目标优于同协议 BC 的幅度 |
| P1 | 旧数据 0/51 完全相同前缀 | 具体哪些 label 被扰动污染 |
| P2 | 部署修正有时极小，方向排序不稳定 | 梯度噪声、后端偏差、状态外推各自占比 |

当前最符合证据的解释不是“少跑了几集”：我们从很少的早期、特定续跑条件下的样本
拟合一个小修正，训练/部署特征还有数值差异，再把它用于新的前缀和长任务；收益与
退化相互抵消。实施修复前需要下面的分层实验，不以新一轮盲目加大 rank/LR/数据量替代诊断。

## 原始证据

远端：`/dev/shm/recap_training_signal_audit_20260909/`。
本地只读副本：`.remote_audit/recap_training_signal_20260909/`。

- `training_signal_v3.json`：604 个训练来源文件的 SHA-256、原始 pair/prefix/动作长度、
  配置、标签、LeRobot schema、12 个 V2 artifact 与 v22 的谱。
- `confirmation_action_audit.json`：仅已完成 confirmation 的动作/配对观测审计。
- `scripts/recap_audit_training_signal.py`：可重复的 CPU-only 审计工具。
- `tests/test_recap_training_signal_audit.py`：初始秩、观测漂移及长度统计测试。

完整诊断 archive 收录 3635 个证据/源码文件，大小 340998536 bytes；
`training_signal_evidence.tar.gz` SHA-256：
`21b8d504caaaebee21efcf1acadb27e48e316b003bdfa761a0d50d732ac6002c`。
`evidence_inventory.json` 和 `persistence.json` 记录逐文件 hash、13 个 adapter 与
当前 frozen plan 的 hash 一致性，以及持久化镜像读回验证。镜像位于
`/vla-cd/ReconVLA/Lingbot-VLA/outputs/recap_training_signal_audit_20260909`。
早期 v1/v2 审计输出也保留；v3 增加了 missing/nonfinite array 的 fail-closed 校验。
本轮回归：本地 **174 passed, 6 skipped**；远端 CPU **179 passed, 1 skipped**，
Ruff 和 diff 检查通过；没有运行新的 GPU kernel/模型/模拟器实验。

当前动作审计还不是完整 failure-stage 诊断；不能据此断言某个接触/抓取阶段的具体物理原因。
没有新增最终集分析方法；没有访问最终 outcome；不修改当前确认门槛或最终 registry。
