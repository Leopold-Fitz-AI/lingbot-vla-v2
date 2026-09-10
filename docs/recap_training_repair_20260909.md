# RECAP 训练修复 P0：实现与执行状态

本轮落实 [三任务修复计划](recap_three_task_repair_plan.md) 的第一道门槛。
**实现/CPU 测试通过不等于完整 6B GPU 对齐通过，更不等于成功率问题已经解决。**
旧 base、adapter、当前确证 study 及其最终分析均不修改。

## 2026-09-10 终端更新

600-input GPU probe 已通过并归档；36-cell P0 重训及 1708-base-tensor 完整性审计
随后全部完成。详见 [冻结设计、全部种子结果及证据位置](recap_training_ablation_20260910.md)。
数值一致和谱退化修复已验证，但旧 holdout surrogate 无一致改善，三个任务成功率修复
未被证明；没有自动启动 P1/P2 或修改生产 registry。下文执行状态保留为 9 月 9 日历史。

## 已实现

### 1. 部署一致的 adapter-only 训练路径

显式 `recap_training_backend: deployment`：

- `prepare_velocity_prefix()` 被采样和新训练共同调用；
- `predict_velocity_features()` 被采样和新训练共同调用；
- `recap_frozen_features()` 暂时把 frozen modules 设为 eval，在 `no_grad` 下计算 h/base v，
  取独立存储的 FP32 张量后恢复每个 module 原来的 training flag；
- **离开 no_grad 才计算 A/B residual 和 loss**。没有把 adapter 的梯度也切掉；
- 固定 teacher-forced x_t 必须不依赖可训练参数。embedding、prompt tuning、可训练
  backbone、完整采样器反传不支持此路径，不能把它当成通用的 no_grad 加速技巧。

此次还确认并处理了训练/部署接口的另一差异：原训练配置为 `flex_cached` 联合
prefix+suffix forward；实际部署为 eager attention、先缓存 prefix 再处理 suffix。
新模式明确要求 eager attention、启用真实 KV cache，而非只把参数转 FP32。

原采样方法提取 helper 前后的计算语句 AST 校验保持一致（基于 commit 454efc9），
并有小 backbone 的共享方法对照。这是代码/CPU 证据；真实 GPU 的 GEMM/cache 行为仍需下一节验证。

### 2. 满秩、同范数、独立 RNG 初始化

`recap_adapter_initialization: orthogonal_matched_v1` 使用独立 CPU generator + float64 QR，
转换到目标 dtype/device。每个 condition 的 A Frobenius 范数匹配旧 CPU-FP32 sine A，
B 仍为零。初始化版本、seed、scale、目标 dtype/device 可追踪。

- rank=8 时八个奇异值非退化，而不是增大 nominal rank 掩盖问题。
- Python、NumPy、Torch 全局 RNG 不改变。
- 旧 `legacy_sin_v1` 保留其原算术，默认仍选 legacy 以保持旧加载和对照实验兼容。
- 新训练应显式同时选 deployment + orthogonal；不能只升级代码却继续默默用旧 recipe。
- 加载已有 artifact 正常覆盖初始化，不重新解释或重写已训练权重。

### 3. 不允许静默改变数值和训练范围

新模式仅支持当前这轮单设备、FP32、micro batch=1、AdamW、无 checkpoint/offload 的路径。
训练入口在 CUDA 初始化前检查环境/配置；运行中检查实际参数/输入和 module 路径：

- 严格 deterministic，不能 warn-only；关闭 autocast/TF32；拒绝编译、分片及可训练输入。
- 只允许 A/B 可训练，base 不得获得 optimizer gradient。
- 禁止视觉辅助目标、router loss/bias 更新；严格模式不注册修改 base buffer 的
  MoE load-balance optimizer hook。
- MoE 记录实际执行的 backend 与 input/expert/output dtype；新路径必须实际走
  `robby_deterministic`，不能仅凭 YAML 写着 FP32。
- 非有限 target、velocity、loss、参数或 optimizer gradient 都在更新前报错。
- FP32 指 tensor dtype；没有声称现有 Triton GEMM 全部改成 IEEE-FP32。

### 4. Artifact 元数据

`scripts/recap_export_adapter.py` 拒绝非有限权重，并把新配置的初始化版本/seed、
训练 backend、config SHA 写入 compact artifact 与 sidecar。
标明是 **configuration provenance，不是 GPU parity 已通过的证明**；尤其是 resume/load
场景中，配置本身不能证明权重在加载后确实按该 scheme reset。

## 600 输入的真实模型门槛

工具：`scripts/recap_audit_backend_parity.py`。

`--prepare` 只用已消费的 V2 原训练/旧 holdout：13 hanging + 20 basket + 17 bowls。
每 state 固定取按路径排序的第一条 manifest，不按 success 选择；完整状态集合必须
与原 split 一致。3 个 flow times × 4 个 noise seeds，共 600 个输入。

- 冻结 manifest/NPZ、3 个原 adapter、wrapper/base shards、norm、Qwen config/tokenizer 和
  executable source 的 SHA；锁有独立 seal，源数据/源码变化即拒绝执行。
- parity 的 x_t 由完整 generated action 归一化构造，是固定输入诊断，不是旧 padding
  训练目标/旧 optimizer steps 的逐字节重放。
- 真正执行 `--run` 前，要求当前冻结 study `stage=complete` 且指定 GPU 没有 compute
  process/模型占用。只允许 CUDA_VISIBLE_DEVICES 显式选一个 GPU 4–7。
- 不等待、抢占、自动重试、启动 simulator、调用 `sample_actions`、训练或更新 registry。
- 运行时比较原训练分支（共同诊断环境下）、新训练 frozen features、实际 server flags
  下的推理接口，以及新的重复计算。记录实际两类 MoE backend。
- 全部 600 个输入的新/推理和重复 h/v 必须 byte-identical；对全部 base tensor 作
  前后 hash 验证。报告 old/new hidden/velocity 漂移与已训练 residual 幅度。
- 输出 `backend_parity.json` 与校验过的逐 state NPZ。异常保留 started/failure，不给
  半个 cohort 下“通过”结论，也不启动后续训练。

## 新训练配置片段

以下是对**独立的新配置副本**的修改，不是完整训练 YAML，不覆盖原配置或 output_dir。
正式 2×2 训练必须先通过上述 GPU gate，并冻结 36 个配置/训练 seed/预算。

```yaml
data:
  recap_enabled: true
  recap_adapter_enabled: true
  recap_prompt_enabled: false
  recap_condition_dropout: 0.0
  image_augment: false
train:
  train_recap_adapter_only: true
  recap_adapter_enabled: true
  recap_adapter_type: velocity_lora
  recap_training_backend: deployment
  recap_adapter_initialization: orthogonal_matched_v1
  recap_adapter_init_seed: 971
  recap_adapter_init_std: 0.02
  recap_adapter_rank: 8
  reset_recap_adapter: true
  attention_implementation: eager
  enable_visual_distillation: false
  sequence_wise_loss_coeff: 0.0
  router_z_loss_coeff: 0.0
  bias_update_speed: 0.0
  enable_fp32: true
  enable_mixed_precision: true  # 在该 trainer 中保证 FP32 parameter construction；不启用 autocast
  enable_full_determinism: true
  enable_gradient_checkpointing: false
  enable_activation_offload: false
  enable_fsdp_offload: false
  use_compile: false
  enable_resume: false
  data_parallel_mode: ddp
  data_parallel_shard_size: 1
  data_parallel_replicate_size: 1
  tensor_parallel_size: 1
  expert_parallel_size: 1
  pipeline_parallel_size: 1
  context_parallel_size: 1
  ulysses_parallel_size: 1
  init_device: cuda
  micro_batch_size: 1
  global_batch_size: 4
  optimizer: adamw
  max_steps: 50
  lr: 0.001
```

其余模型结构、动作归一化、数据路径、原 objective、scale、loss、scheduler 不同时改。
原 4-GPU/FSDP recipe 不能直接和新单设备 recipe 归因比较：2×2 的四个条件都应使用
同一单设备执行框架，并明确这是新的控制实验，不是原训练的精确复现。

## 当前状态和后续

- 2026-09-09 08:18 UTC 的 outcome-blind 检查：GPUs 4–7 各有模型占用，冻结 controller
  存活；最终 primary 已 job-validated 1000，另 578 recorded/unvalidated。
  **本轮不据此计算任何成功率，也未启动 GPU probe 或新训练。**
- P0 实现与 CPU 测试已完成：本地 **231 passed, 6 skipped**；隔离远端代码树
  **236 passed, 1 CUDA-only module skipped**。没有运行新的 GPU kernel/model 测试。
- 已成功冻结并核验 **50 states / 600 inputs、300 个 executable source、127 个输入文件**。
  实际 CPU 准备发现 wrapper 的 norm_stats_file 为 None；已让审计器遵循真实
  robot_config fallback，并补充回归测试。这是审计器准备逻辑的修正，**不是改变模型归一化**。
  初次失败的 CPU 准备日志保留，未创建 input lock 或启动 GPU 模型。
- **2026-09-09 09:37 UTC** 已部署 tmux `recap-p0-probe` 单次等待器，实际状态为
  `waiting_for_frozen_study`；没有 `started.json`，尚未运行 6B 模型。
  源码在 `/dev/shm/recap_training_repair_20260909/code`，输入锁/队列在同 root 的
  `backend_probe/`，与正在执行的 study/code 和 mutable remote source 都分离。
- 完整 6B GPU parity、36 次 2×2 重训、成功率收益仍未验证。probe 只在 GPU 4
  比较同输入/同设备，不能据此给出全硬件 bitwise 一致保证。
- 新多阶段/多 continuation 数据、70 对失败阶段标注、pairwise objective、retention/gate
  仍属于后续 P1；不能把当前代码修复说成已经补齐了训练数据或解决了所有泛化问题。
- 当前最终实验仍按原计划独立完成并报告，绝不与后续实验拼接凑显著。

另有 `scripts/recap_wait_backend_probe.py`：只做 outcome-blind 资源等待，最多 48 小时。
当前 study 无论最终为 significant_uplift 还是 not_proven，都使用同一启动规则；
基础设施失败则停止等待并要求人工审计。当前实验完整终止且 GPU 4 空闲后，
exclusive launch latch 只允许启动一次 probe，任何退出都不重试，也不自动训练。
它不重启 study/controller，不读取 `final_result.json`。
probe 结束后仍须审核实际 `backend_parity.json`/失败证据并对 terminal 输出作持久化读回
验证；排队或 process exit 本身不是 P0 通过，也不会自动推进到重训或生产路由。

在独立修复 snapshot 中**手动**运行 probe 的命令（已排队时不要再手动启动）：

```bash
cd /dev/shm/recap_training_repair_20260909/code
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=4 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0 \
/dev/shm/conda-lingbot-recap/bin/python scripts/recap_audit_backend_parity.py \
  --run --gpu 4 \
  --active-study /dev/shm/recap_confirmatory_20260908_locked_cohorts \
  --output /dev/shm/recap_training_repair_20260909/backend_probe
```
