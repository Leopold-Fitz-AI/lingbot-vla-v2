# 现有 P0 修复模型的小规模闭环检查结果

## 结论

2026-09-11 04:55 UTC，一次性探索检查完整结束：**120 个新配对环境状态、240 policy
episodes**，无整任务重试。现有 P0 正式模型没有显示值得直接扩大验证的闭环信号。
这不是“模型确定无效”的证明；每任务只有 40 对，区间仍宽。

| 任务 | 修复模型 | Base | rescued / regressed | delta | exact paired p |
|---|---:|---:|---:|---:|---:|
| hanging_mug | 20/40 | 21/40 | 3 / 4 | −2.5 pp | 1.0 |
| place_can_basket | 33/40 | 33/40 | 4 / 4 | 0 pp | 1.0 |
| stack_bowls_three | 33/40 | 34/40 | 5 / 6 | −2.5 pp | 1.0 |

三任务等权 macro delta：**−1.67 pp**；固定 20,000 次 task-stratified paired bootstrap
95% CI：**[−10.0, +6.67] pp**。三个任务 Holm p 均为 1.0。

预注册的“继续研究优先级”旗标要求 macro ≥+5 pp 且每任务无净退化，结果为 **false**。
该旗标不是显著性检验；false 也不等于真实效应必为零或负值。

## 固定设计

协议见 [`recap_repaired_policy_check_20260911.md`](recap_repaired_policy_check_20260911.md)。

- 只测已存在的 P0 `deployment + orthogonal_matched_v1 + optimizer seed 2101` artifacts；
  没有新训练或 best-seed 选择。
- 沿用此前已定窗口：挂杯 all；篮子、碗 d1-only。
- 新且预先审计未使用的 index 120，每任务首 40 个专家可行状态；Null 先于 repaired。
- 挂杯/篮子/碗固定 GPU 5/6/7。GPU 0–3 未使用。
- 每个 task×arm 只有 `attempt-1`，无 incomplete cohort、替换状态或 score-dependent retry。
- 锁定 cohort、literal instruction、初态 hash 和严格 FP32/eager deterministic runtime 均由原审计执行器验证。
- 未访问旧 final inputs；没有 P1 pilot、采集器、额外训练、新 50-task final 或 production 更新。

## 解释边界

P0 已证明的数值事实——训练/部署冻结特征一致、满秩初始化和 base tensors 不变——没有转化为
本次可见的任务收益。因此下一步不应扩大验证，也不应仅通过提高 rank、LR 或 steps 追分。
更窄的合理路线是：选择**一个任务和一个阶段数据覆盖假设**，在任何运行前另行冻结；当前
结果本身不授权自动开始。

本检查不能拆分 backend 与初始化的闭环因果贡献，因为只评估一个预先固定配置；也不能用于
50-task suite 的显著性或官方结果比较。

## 完整性与证据

权威 root：`/dev/shm/recap_repaired_policy_check_20260911`；持久化镜像：
`/vla-cd/ReconVLA/Lingbot-VLA/outputs/recap_repaired_policy_check_20260911`。

- `result.json`：冻结汇总；`paired_outcomes.json`：全部 120 对结果。
- `completion_audit.json`：独立只读复算并验证 6 个完整 jobs、全部 cohort/initial hashes、
  authoritative manifest/NPZ 与 NFS read-back；observer 运行 0 policy episodes。
- 启动前测试：远端 28 passed；本地完整回归 263 passed / 6 skipped。
- `supervisor_launch_error.json` 是启动 marker 尚在生成时被误判而留下的记录；没有覆盖。
  `supervisor_record_correction.json` 明确记录唯一 controller PID，未发生第二次 controller/policy launch。
- 终端证据 archive：`terminal_evidence_v1.tar.gz`，SHA-256
  `661998c95ac766f2806ee64cae1893fda638c56102484ce3f785a1fb1ac26ce3`；
  root 与持久化镜像的 `terminal_persistence.json` 记录 SHA/read-back 清单。
