# P0 完整模型优化归因实验 — 2026-09-10

用户在 2026-09-10 要求继续完成剩余任务后执行。独立于已经完成的确证实验。
[原最终结果](recap_final_result_20260910.md) 只作完整报告，不反馈到训练/调参。
旧 base、adapter、失败实验、frozen source、当前生产 registry 均不修改。

## 前置证据已完成

- 原最终研究：6000 primary + 60 Negative 完整；冻结统计和所有镜像由独立 observer
  复算通过。套件均值 92.8667% → 93.1333%，+0.2667 pp，CI [0.1000, 0.4333]，
  p=.0078125；8 rescued / 0 regressed，仅 click_bell 路由了 adapter。
  这是已冻结策略的固定套件均值提升，不是三个待修复任务已改善。
- 2026-09-10 00:32 UTC，GPU 4 的 600 输入 probe 通过，1708 base tensor 不变。
  新训练 frozen features / inference velocity 及重复 h/v 全部字节一致。
- Probe 的全部 59 个 terminal 文件已 SHA/read-back 核验保存于：
  `/vla-cd/ReconVLA/Lingbot-VLA/outputs/recap_training_repair_20260909/backend_probe_terminal`。
  收据：`/dev/shm/recap_training_repair_20260909/terminal_persistence_v1.json`。

| 已消费固定输入 | points | 旧/新 velocity RMS 均值 | 当前旧 adapter residual RMS 均值 |
|---|---:|---:|---:|
| hanging_mug | 156 | .00064179 | .00026941 |
| place_can_basket | 240 | .00091702 | .00015788 |
| stack_bowls_three | 204 | .00061988 | .00212225 |

以上不同量的比值不是建议的 action scale；不是误差导致成功率下降的因果归因。
GPU 4 的相同输入结果也不等于所有硬件都保证 bitwise 一致。

## 固定设计

`/dev/shm/recap_ablation_20260910/study/plan.json` 和 `lock.json` 在训练前建立。
代码位于相邻 `code/`，与上一个 parity snapshot/最终研究都隔离。
302 个 executable source、350 个输入文件参与 hash 检查。

- 3 tasks × 2 backend × 2 initializer × 3 optimizer seeds = **36 cells**。
- optimizer seeds：**2101, 2102, 2103**；正式 seed 固定为 **2101**，不能择优换 seed。
- 初始化私有 seed：**971**；rank 8，scale 8，init_std .02；新旧初始化同范数。
- 每 cell 50 optimizer steps，global batch 4，micro batch 1，lr .001；
  AdamW betas (.9,.95)、eps 1e-8、weight_decay 0、clip norm 1、constant LR。
- 总 **1800 optimizer updates / 7200 sample presentations**，不是新环境状态。
- Hanging：signed，26 chunks，GPU 5；basket：signed + residual L2(100)，40 chunks，GPU 6；
  bowls：positive-only，18 chunks，GPU 7。只使用原来的训练集。
- 所有 cells 均为单设备 FP32、matmul highest、TF32 off、无辅助目标/checkpoint/offload/
  router buffer 更新。旧 backend 是 fused BF16 MoE + joint/flex_cached forward；新 backend
  是 strict deterministic / eager / real prefix cache。
- 采用 PCG64 每 epoch permutation + drop_last；固定完整 50 batches。相同 task/seed 的
  四个 cells 共享 row indices 和逐 step/micro 的显式 noise/t draws，hash 必须一致。
  这是统一单设备框架的新归因实验，**不是原 4-GPU FSDP 或原 main() 的精确复现**。
- **0 policy episodes**，不调用模拟器或 sample_actions，不访问最终环境输入/成功结果。

## 真实训练入口、缓存与门槛

工具：`scripts/recap_run_training_ablation.py`、`lingbotvla/recap/ablation.py`。
使用原 LeRobot dataset 与实际 `VLADataset/FeatureTransform`。冻结前 CPU 已验证：
26/40/18 个原训练 rows 与原始 NPZ 重建的全部模型输入/目标/mask/condition 字节一致。
不是手写 14→55 action mapping，也没有重新监督 padding。

每 task/seed/backend 的 sine cell 运行真正的 `LingbotVlaV2Policy.forward` 和 backward。
orthogonal cell 复用同输入的 detached h/v，因为这两个 A/B 仅位于冻结 backbone 之后，
不能改变 teacher-forced x_t 或 h。两者有独立 A/B、独立 AdamW state、各自 50 次更新。

- 每个 microbatch：缓存公式的 loss 必须与 production forward 精确相等。
- 前两步每 microbatch：缓存 A/B gradients 必须与 production autograd 精确相等。
- B=0 的 Null/Positive/Negative 与 base 一致；第一步 B 非零梯度/A 零梯度；
  第二步 A/B 都有非零梯度；每次更新后 Null 仍精确零 residual。
- 所有 base gradients 必须为 None。每组前后验证全部 1708 base tensor hashes。
- 实际 MoE backend telemetry 必须与声明一致；非有限 loss、参数或梯度直接停止。
- 每 cell 保存 compact safetensors + 完整 recipe/初始化/采样/update 日志，做 reload/hash
  验证，不写出 36 份重复的 6B base，也不覆写已有 adapter。
- 任一 worker 失败写 stop；其他已运行的组收尾后不再开始新组。无自动重试/恢复。

## 后验诊断的范围

三任务原训练/旧 holdout 共 148 个 paired action chunks，分别 38/54/56。
每 chunk 固定 3 t × 4 noise，共 1776 个 strict frozen-feature inputs。
所有 12 cells/task 都在相同 deployment features 上计算正条件 FM error。

报告 `S(a)=E_base(a)-E_adapter(a)` 的 pair margin、正样本 BC 变化，按 train/旧 holdout
分列。旧 pair 的原观测漂移仍按历史事实保留，没有称其成为新的精确 causal pair。
多个 noise draws 不算多个独立状态；这些 loss/surrogate 指标不证明真实成功率提升。

全部 cells 完成后，独立 `scripts/recap_finalize_training_ablation.py` 验证完整 36-cell
预算、相同 draw schedules、base/adapter hashes、原诊断重算及逐 cell 的 A/BA 谱，
生成 `reports/initialization_ablation.json`。它不挑模型、不中途汇总成功率、不启动下一阶段。
终端输出全部持久化并读回校验；异常同样保留，不分析不完整 cohort 为“完成”。

## 完成结果 — 2026-09-10 03:35 UTC

**36/36 cells、1800/1800 optimizer updates 全部完成；无重试、无 policy episode。**
三个 worker 在 03:33–03:34 UTC 写出 done；03:35:24 UTC 独立 observer 完整审计通过。
全部 1708 base tensor 不变，36 份 compact adapter 均保留，没有择优替换正式 seed 2101。
终端报告和 111 个输出文件已 SHA/read-back 持久化到：
`/vla-cd/ReconVLA/Lingbot-VLA/outputs/recap_ablation_20260910/study`。
报告为 `reports/initialization_ablation.json`，收据为 `reports/persistence.json`。
另外保存了完整 302-file sealed executable archive 和 120-file terminal/log archive，
均经归档前后原文件 hash、tar 成员 hash、NFS 及本地下载 read-back 核验。

| archive | SHA-256 |
|---|---|
| `source_evidence_v1.tar.gz` | `7008c4edfc87775218f78b48808638a10a1522aab0c50d9aa067fbdee66eae37` |
| `terminal_evidence_v1.tar.gz` | `699f7006ccce5030a8d24dd364e1d84324f2a64ba2f0f4635b58a3f51e0465e2` |

收据：`terminal_archive_persistence_v1.json`。302 个 executable 与提交源码逐文件相符。
最终本地 RECAP regression（含独立 observer 和 blind-pack tests）：**248 passed / 6 skipped**；
运行前远端 sealed worker regression 为 **246 passed / 1 skipped**。

### 全部 optimizer seeds 的旧 holdout 配对 surrogate margin

下表单位 **1e-6 flow-error**，不是成功率或 pp。正数表示正条件对 good chunk 的
相对 FM 改善大于对 bad chunk 的改善；这不是 Q、log likelihood 或成功率估计。
每格依次列 seed 2101 / 2102 / 2103，括号内为三 seed 均值。没有选择最佳 seed。

| task | old backend / sine | old backend / orthogonal | deployment / sine | deployment / orthogonal |
|---|---|---|---|---|
| hanging | 16.59 / −25.49 / 6.48 (−0.80) | 25.87 / −11.87 / −0.47 (4.51) | 11.29 / −25.97 / 10.08 (−1.53) | 22.78 / −8.72 / −0.61 (4.48) |
| basket | −2.58 / 2.19 / 3.36 (0.99) | 1.47 / 2.79 / −0.39 (1.29) | −3.51 / 4.77 / 3.89 (1.71) | 3.65 / −3.30 / −2.56 (−0.74) |
| bowls | −29.91 / −92.63 / −49.34 (−57.29) | −43.46 / −76.34 / −55.26 (−58.36) | −38.53 / −82.35 / −45.66 (−55.51) | −50.08 / −91.69 / −55.22 (−65.67) |

**没有观察到跨任务/种子一致的 surrogate 改善。** Bowls 的 positive-only BC 本来就没有
配对 margin 项；负 margin 不能被解释成 Negative policy 成功率更高，也不能据此切换方向。
已有数据的窄阶段覆盖、单次未来 outcome 标签和缺少 paired/retention 目标仍未被这两项
数值/初始化修复解决。不能宣称三任务闭环泛化已修复。

初始化的结构性变化确实保留下来了：正式 seed 2101、deployment backend 的训练后 BA
前两奇异方向能量占比，sine → orthogonal 分别为 hanging **99.575% → 57.537%**，
basket **99.572% → 73.871%**，bowls **99.586% → 87.224%**。
这是矩阵谱证据，不是成功率的因果效应。

### 后续轨迹证据包，不冒充已完成阶段标注

`scripts/recap_pack_failure_stages.py` 已导出 70 个完整 confirmation discordant pairs
和 30 个按 seed 预定 tie controls，共 100 pairs / 200 张图（storyboard + raw command
joint/gripper timeline）。只读取 indices 43–44，并核对原 done 中的 manifest/NPZ SHA。
位置：`/dev/shm/recap_failure_stage_audit_20260910`；对应 outputs 镜像中的
`unreviewed_evidence_v1.tar.gz` 包含 204 个文件（含实际生成脚本），SHA
`dfba609a758e6f6320baa25c6d200799285c52158e30728db0534e47d131e230`。
NFS 与本地 archive hash 已核验。
公开 A/B 顺序及案例排列与 condition/outcome 分离；`unblind_mapping.json` 单独保存。
`review_template.json` 中语义阶段仍标为 **unreviewed**；初始图像抽查不是完整人工阶段审计。
稀疏观测在每个 action chunk 前记录，不能把最后一张图当作执行最后 chunk 后的终态，
也不能凭终局 success 或图像猜接触力。正式 `failure_stage_audit.json` 尚未生成。

P1 的 1936-rollout causal pilot、45 个新 objective trainings、P2 新独立验证仍未启动或排队；
须先完成阶段审计、冻结可观测 anchor/schema、检查并预留未消费 seed IDs。
不直接将 P0 artifacts 放进生产 registry 或旧最终研究。

## 历史启动状态

- 2026-09-10 **03:12 UTC**，三个训练 worker 实际启动；GPUs 0–3 未使用，GPU 4 空闲。
- tmux：`recap-p0-train-5`、`recap-p0-train-6`、`recap-p0-train-7`。
- 日志：`/dev/shm/recap_ablation_20260910/train_gpu{5,6,7}.log`。
- CPU regression：原运行代码远端 246 passed / 1 CUDA-only skipped；独立 observer 5 passed。
- 终端 observer：tmux `recap-p0-ablation-report`；日志同 root 的 `observer.log`。
- **以实际 done/report 为准，启动/梯度通过不代表重训或泛化修复全部完成。**
- 后续仍需完成 condition-blind failure-stage audit、多阶段/shared-continuation pilot、
  schema/pair objective 与保持能力实验。P0 输出不是自动进入新正式验证或生产 registry 的许可。
