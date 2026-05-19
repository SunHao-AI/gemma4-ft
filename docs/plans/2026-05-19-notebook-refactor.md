# Notebook Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 抽取四个 notebook 中的重复公共代码，沉淀到 `notebooks` 目录模块，并在不改变功能的前提下完成导入式重构与可运行性验证。

**Architecture:** 以 `notebooks.common` 继续负责 notebook 上下文初始化；新增视觉推理、评估展示等模块承载 `03/04` 的共享逻辑；对 `02` 仅抽取稳定、可跨 notebook 复用的路径/训练输出解析辅助函数，避免过度抽象。

**Tech Stack:** Python, Jupyter Notebook (`.ipynb` JSON), Unsloth, PEFT, PyTorch, Matplotlib, PIL

---

### Task 1: 重复逻辑梳理

**Files:**
- Modify: `docs/plans/2026-05-19-notebook-refactor.md`
- Review: `notebooks/01-data_preparation-labelme_processing.ipynb`
- Review: `notebooks/02-model_finetuning.ipynb`
- Review: `notebooks/03-object_detection_demo.ipynb`
- Review: `notebooks/04-model_comparison.ipynb`

**Step 1: 标记共享区域**

- notebook 启动/上下文初始化
- Gemma4 + LoRA 模型加载
- 目标检测 prompt 构造、输入组装、结果解析
- 图片加载与检测可视化
- 评估结果表格/差异分析/JSONL 数据解析

**Step 2: 确定模块边界**

- `notebooks.common`: 保留上下文初始化
- `notebooks.vision_shared`: 提供模型加载、图像加载、检测器、可视化器
- `notebooks.eval_shared`: 提供评估数据加载、结果表格、分析报告等
- `notebooks.train_shared`: 提供训练输出目录解析等轻量辅助函数

### Task 2: 提取公共模块

**Files:**
- Create: `notebooks/vision_shared.py`
- Create: `notebooks/eval_shared.py`
- Create: `notebooks/train_shared.py`

**Step 1: 抽取稳定共享代码**

- 从 `03/04` 提取 `ModelLoader`
- 抽取图像加载、检测 prompt 与 `box_2d` 解析
- 抽取可视化基础能力，保留 notebook 侧差异化展示组合
- 抽取数据集 JSONL 解析、表格格式化、分析报告
- 抽取训练产物目录解析辅助函数

**Step 2: 保持 API 简单**

- 尽量兼容 notebook 原变量命名
- 避免把 notebook 展示逻辑过度塞进公共模块
- 优先抽取无状态工具函数和通用类

### Task 3: 回写四个 Notebook

**Files:**
- Modify: `notebooks/01-data_preparation-labelme_processing.ipynb`
- Modify: `notebooks/02-model_finetuning.ipynb`
- Modify: `notebooks/03-object_detection_demo.ipynb`
- Modify: `notebooks/04-model_comparison.ipynb`

**Step 1: 替换重复实现**

- 删除被抽取的 class / function 定义
- 改为从新模块导入
- 保留每个 notebook 的配置单元、演示流程与输出结构

**Step 2: 适配差异化调用**

- `03` 使用共享检测与展示类构建单模型检测流程
- `04` 使用共享检测器 + 评估/展示工具构建对比分析流程
- `02` 使用训练共享辅助函数简化路径/输出目录相关逻辑
- `01` 优先复用已有 `labelme_tools`，仅整理 notebook 内部残留重复工具

### Task 4: 验证与收尾

**Files:**
- Test: `notebooks/01-data_preparation-labelme_processing.ipynb`
- Test: `notebooks/02-model_finetuning.ipynb`
- Test: `notebooks/03-object_detection_demo.ipynb`
- Test: `notebooks/04-model_comparison.ipynb`

**Step 1: 运行静态检查**

- 解析四个 notebook JSON 结构
- 验证新增导入路径有效
- 对新增 `.py` 模块运行基础语法检查

**Step 2: 运行轻量 notebook 验证**

- 逐个执行导入/配置相关代码单元
- 验证 `03/04` 的共享模块可正确导入
- 验证 `02` 的训练输出辅助函数和评估配置仍可构建
- 使用可行的 smoke test 替代超重训练/长时推理全量重跑
