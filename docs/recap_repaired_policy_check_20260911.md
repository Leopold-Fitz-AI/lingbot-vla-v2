# 现有三任务修复模型：一次性小规模闭环检查

## 问题与边界

只回答：P0 已经训练完成的正式模型在新环境状态上，是否显示值得继续投入的真实闭环信号。
本检查不是生产验收、不是各项工程修复的因果归因，也不要求一次小样本得到显著性。

- 固定任务：`hanging_mug`、`place_can_basket`、`stack_bowls_three`。
- 固定模型：P0 `deployment/orthogonal_matched_v1`，正式 optimizer seed **2101**。
  不查看/选择 2102、2103 的闭环结果，也不再训练。
- 固定窗口沿用旧 screen 在任何 P0 闭环结果出现前的选择：挂杯 all-window；篮子和碗 d1-only。
- 比较：修复模型 Positive 与空 registry/base；相同任务在同一 GPU，使用锁定 cohort、相同
  environment seed、instruction、初态 hash、policy/noise seed。
- 新 index **120**，每任务 **40** 个专家可行状态，每 arm 40；共 120 个独立配对状态、
  **240 policy episodes**（不含有界基础设施整任务重试及专家预检）。
- 任务固定 GPU：挂杯 5、篮子 6、碗 7。GPU 0–3 禁用，GPU 4 留空。
- 先跑 Null，再跑 repaired；不因中间成功率停跑、换状态、换 seed、改窗口或改模型。
- 沿用冻结确定性协议：policy seed 900、continuation 300、counterfactual decision 0、
  per-episode common noise、literal deterministic instruction、chunk 50、CFG 1、FP32/eager、
  deterministic custom MoE。初态必须与预检精确 hash 一致；协议错误退出 78，不重试。
- 使用原 launcher 的每任务最多三次基础设施尝试和每状态六次物理 setup 上限；耗尽则整项
  标记不完整，不用已完成子集作结论。
- 不访问旧 final inputs；旧 indices 42–44 和 50–52 不进入本次新 cohort。

## 预先固定的汇总

每任务完整报告 Positive/Base 成功数、rescued、regressed、净 delta、双侧 exact paired p，
并给三个任务 Holm 调整值。另报告三任务等权 macro delta 和固定 RNG 20260911、20,000 次
按 task 分层的 paired bootstrap 95% CI。

仅用于排下一步优先级的旗标（不是显著性或部署门槛）：

1. 三任务等权 macro delta 至少 **+5 pp**（120 对合计净救回至少 6）；且
2. 三个任务均无净退化。

满足只表示“值得设计独立确认”；不满足也可能因 N=40/任务而不确定，不能证明无收益。
无论结果如何，脚本都不会自动训练、启动采集 pilot、扩大验证或修改 production registry。

## 一次性执行与完整性

运行前冻结源码、plan、模型 SHA、unused-index 审计和资源检查。`launch_once.json` 使用排他创建；
同一 root 不允许覆盖、重启或 outcome-dependent retry。所有失败和部分证据保留。只有 3×40×2
完整、runtime/registry/cohort/初态验证通过后才生成 `result.json`。
