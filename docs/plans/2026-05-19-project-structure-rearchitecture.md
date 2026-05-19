# Project Structure Rearchitecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 基于当前 Gemma4 多模态训练仓库完成一次工程化目录重构，将源码、脚本、实验材料与文档职责分层，并在保证现有流程可运行的前提下完成路径迁移与兼容更新。

**Architecture:** 采用主流 Python/ML 应用型项目结构：源码统一收口到 `src/gemma4_ft/`，按 `core/data/training/tools/notebooking` 分层；`notebooks/` 仅保留 `.ipynb` 实验与展示资产；根级旧包保留为极薄兼容层，避免历史导入与外部脚本立即失效。通过更新 `pyproject.toml`、脚本入口、文档路径与测试导入，逐步将仓库切换到新的主包结构。

**Tech Stack:** Python 3.10+, setuptools/pyproject, PyTorch, Transformers, Unsloth, pytest, Jupyter Notebook, Docker

---

### Task 1: 输出问题分析与目标结构

**Files:**
- Create: `docs/project-structure-review.md`
- Review: `ARCHITECTURE.md`
- Review: `REFACTOR.md`
- Review: `pyproject.toml`

**Step 1: 汇总当前结构问题**

梳理当前仓库在以下维度的问题：
- 顶层目录平铺，源码职责边界不清
- notebook 辅助模块与 `.ipynb` 混放
- 训练脚本同时承担“包模块”和“裸脚本”双重职责
- 根级文档与模块文档存在失配
- 缺少统一源码根与推荐入口

**Step 2: 写入问题分析文档**

在 `docs/project-structure-review.md` 中输出：
- 当前问题清单
- 现有目录职责评估
- 对照主流 Python/ML 工程结构的差距
- 新结构设计目标与迁移原则

**Step 3: 定义目标目录树**

文档中给出目标结构：

```text
src/gemma4_ft/
  core/
  data/labelme/
  training/distributed/
  tools/color_contrast/
  notebooking/
scripts/
notebooks/
tests/
docs/
docker/
```

**Step 4: 记录迁移映射**

明确以下映射：
- `gemma4_core/* -> src/gemma4_ft/core/*`
- `labelme_tools/* -> src/gemma4_ft/data/labelme/*`
- `distributed_training/* -> src/gemma4_ft/training/distributed/*`
- `color_contrast_tools/* -> src/gemma4_ft/tools/color_contrast/*`
- `notebooks/*.py -> src/gemma4_ft/notebooking/*`
- 根级执行脚本通过 `scripts/` 或兼容层指向新模块

### Task 2: 建立新源码结构并迁移包实现

**Files:**
- Create: `src/gemma4_ft/__init__.py`
- Create: `src/gemma4_ft/core/*.py`
- Create: `src/gemma4_ft/data/labelme/*.py`
- Create: `src/gemma4_ft/training/distributed/*.py`
- Create: `src/gemma4_ft/tools/color_contrast/*.py`
- Create: `src/gemma4_ft/notebooking/*.py`
- Modify: `pyproject.toml`

**Step 1: 创建新主包骨架**

创建 `src/gemma4_ft/` 及分层子包，补齐 `__init__.py`。

**Step 2: 迁移实际源码**

将现有实现迁移到新结构，优先保持文件内容稳定，只调整：
- 包路径
- 绝对导入
- 少量项目根解析逻辑

**Step 3: 更新主包导出**

在新包 `__init__.py` 与各子包 `__init__.py` 中提供清晰导出，作为新的标准导入入口。

**Step 4: 切换打包配置**

修改 `pyproject.toml`，启用 `src` 布局包发现，使 `gemma4_ft` 成为标准安装入口。

### Task 3: 建立兼容层并更新路径引用

**Files:**
- Modify: `gemma4_core/__init__.py`
- Modify: `gemma4_core/*.py`
- Modify: `labelme_tools/__init__.py`
- Modify: `labelme_tools/*.py`
- Modify: `distributed_training/__init__.py`
- Modify: `distributed_training/*.py`
- Modify: `color_contrast_tools/__init__.py`
- Modify: `color_contrast_tools/*.py`
- Modify: `notebooks/*.ipynb`
- Modify: `notebooks/common.py`
- Modify: `docker/run_train.sh`
- Modify: `docker/run_infer.sh`
- Modify: `docker/Dockerfile`
- Modify: `docker-compose.yml`

**Step 1: 旧包改为兼容层**

将根级旧包改造成兼容层，转发到新主包，避免：
- 测试立即失效
- 现有 notebook/脚本路径立即失效
- 文档中旧示例完全不可运行

**Step 2: 更新标准导入路径**

把项目内部标准导入切换到：
- `gemma4_ft.core.*`
- `gemma4_ft.data.labelme.*`
- `gemma4_ft.training.distributed.*`
- `gemma4_ft.tools.color_contrast.*`
- `gemma4_ft.notebooking.*`

**Step 3: 更新 notebook 与脚本路径**

更新 notebook 内共享模块导入和训练/推理脚本路径引用，确保：
- `.ipynb` 只保留实验流程
- 共享 notebook helper 不再与实验文件混放

**Step 4: 更新容器与配置路径**

同步更新 Docker、compose、shell 脚本和文档中的关键路径。

### Task 4: 验证与文档交付

**Files:**
- Modify: `ARCHITECTURE.md`
- Modify: `README.md`
- Create: `docs/project-structure-guide.md`
- Test: `tests/*.py`

**Step 1: 运行静态与编译校验**

执行：
- `python -m compileall src tests notebooks`
- `pytest` 关键测试集

**Step 2: 验证入口可用**

验证：
- 新主包导入可用
- 兼容层导入可用
- 训练/推理入口路径无模块缺失

**Step 3: 更新仓库级说明**

更新 `README.md` 和 `ARCHITECTURE.md`，明确：
- 新目录树
- 模块职责
- 推荐开发入口
- 兼容层说明

**Step 4: 交付结构说明文档**

在 `docs/project-structure-guide.md` 中输出：
- 新目录结构说明
- 资源存放规范
- 模块依赖关系
- 团队协作约定
- 后续可逐步删除的兼容层清单
