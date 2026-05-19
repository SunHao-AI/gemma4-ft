# Multi-GPU Load Balance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为当前 Gemma4 多 GPU 推理链路增加动态负载均衡调度、GPU 状态监控和性能对比报告能力。

**Architecture:** 复用现有 `torchrun -> distributed_inference.py` 多进程入口，不新增旁路服务；新增一个可单测的 SQLite 任务队列/性能分析模块，供 worker 进程以“谁空闲谁拉取”方式动态认领任务。Notebook 仅负责暴露调度模式配置并回收新增报告，不改变现有结果展示与指标分析结构。

**Tech Stack:** Python, torch.distributed, sqlite3, Jupyter Notebook JSON, pytest

---

### Task 1: 梳理入口与约束

**Files:**
- Modify: `distributed_training/distributed_inference.py`
- Modify: `notebooks/04-model_comparison.ipynb`
- Test: `tests/test_load_balancer.py`

**Step 1: 识别现有静态分片、worker 统计、Notebook 参数透传入口。**

**Step 2: 确认新增调度模式命名与结果文件落盘位置。**

**Step 3: 确认不会破坏现有 `single` / `multi_gpu` 的结果回收路径。**

### Task 2: 实现动态队列模块

**Files:**
- Create: `distributed_training/load_balancer.py`
- Test: `tests/test_load_balancer.py`

**Step 1: 编写任务队列初始化、原子 claim、完成回写与 worker 心跳更新测试。**

**Step 2: 实现 SQLite 任务队列、GPU 指标采集和负载分析函数。**

**Step 3: 编写调度仿真与报告渲染测试，确保动态策略优于静态尾部负载。**

### Task 3: 接入分布式推理脚本

**Files:**
- Modify: `distributed_training/distributed_inference.py`
- Test: `tests/test_load_balancer.py`

**Step 1: 新增 `scheduler_mode` 与动态队列参数解析。**

**Step 2: 保持静态分片逻辑不变，同时新增动态队列轮次执行逻辑。**

**Step 3: 将 worker 状态、GPU 监控、性能摘要与 Markdown 报告一并写入 `result_dir`。**

### Task 4: 接入 Notebook 与报告展示

**Files:**
- Modify: `notebooks/04-model_comparison.ipynb`

**Step 1: 将多 GPU 配置升级为“调度模式 + 静态分片策略”。**

**Step 2: 在 torchrun 命令生成中透传新参数。**

**Step 3: 在多 GPU 结果回收阶段打印负载均衡报告路径和关键摘要。**

### Task 5: 验证与交付

**Files:**
- Modify: `docs/reports/2026-05-18-multi-gpu-load-balance.md`
- Test: `tests/test_load_balancer.py`

**Step 1: 运行目标单测与诊断检查。**

**Step 2: 生成实现说明、测试结果和性能对比分析文档。**

**Step 3: 记录未在本地完成的真实多 GPU 实测前提与执行方式。**
