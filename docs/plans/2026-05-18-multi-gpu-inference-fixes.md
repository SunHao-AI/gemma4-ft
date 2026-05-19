# Multi GPU Inference Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复多 GPU 推理中的显存负载不均、轮次切换校验缺失与日志进度展示混乱问题。

**Architecture:** 继续复用 `torchrun -> distributed_training/distributed_inference.py` 的现有链路，在 worker 侧补充严格设备映射校验、任务/结果完成检查与统一进度日志；在 `load_balancer.py` 中补充队列进度/完成态查询；在 Notebook 中调整默认 torchrun 日志策略。

**Tech Stack:** Python, torch.distributed, Unsloth, SQLite, Jupyter Notebook

---

### Task 1: 排查并固化根因

**Files:**
- Modify: `distributed_training/distributed_inference.py`
- Modify: `notebooks/04-model_comparison.ipynb`
- Reference: `temp.log`

**Step 1: 核对日志与 worker 绑定关系**

- 确认 `temp.log` 中的 rank/gpu 关系、轮次切换点和进度输出模式。

**Step 2: 识别 Notebook 与 worker 的职责边界**

- 确认 `multi_gpu` 模式下 Notebook 不会额外预加载模型，问题应收敛到 worker。

### Task 2: 修复 worker 设备绑定与轮次切换

**Files:**
- Modify: `distributed_training/distributed_inference.py`
- Modify: `distributed_training/load_balancer.py`

**Step 1: 增加严格设备映射配置与校验**

- 用显式单卡 `device_map` 替代宽松字符串映射。
- 在模型加载后校验 `hf_device_map` / 参数设备，发现越界映射立即报错。

**Step 2: 增加 round 完成校验**

- 在 round 1 结束后验证动态队列已全部完成。
- 在进入 round 2 前验证 partial 结果文件数量和样本总量符合预期。

**Step 3: 精简进度日志**

- 禁用 live tqdm 默认输出。
- 仅保留 rank0 的统一核心统计信息，如“总数/已处理/成功/失败/剩余”。

### Task 3: 验证并记录结果

**Files:**
- Modify: `tests/test_load_balancer.py`
- Modify: `notebooks/04-model_comparison.ipynb`

**Step 1: 补充队列完成态测试**

- 为进度快照与完成校验增加单测。

**Step 2: 运行静态验证**

- 执行 `GetDiagnostics`、`py_compile`、`pytest tests\test_load_balancer.py -q`。

**Step 3: 记录真实运行限制**

- 若当前环境无目标多 GPU 运行条件，明确说明无法在本地完成端到端复跑，并给出建议复跑命令与验证点。
