# Docker Training Environment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 Gemma4 项目补齐可复用的 Docker 训练与推理环境，使 CUDA toolkit、nvcc、conda 与 flash-attn 都在容器内管理。

**Architecture:** 采用 `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` 作为基础镜像，在镜像中安装 Miniforge、Python 3.11、PyTorch cu128、项目依赖与 flash-attn。通过 `docker-compose.yml`、入口脚本和操作文档把宿主机准备、镜像构建、容器内训练与推理解耦。

**Tech Stack:** Docker, Docker Compose, NVIDIA Container Toolkit, CUDA 12.8, Miniforge, Python 3.11, PyTorch 2.10, Unsloth, flash-attn

---

### Task 1: 宿主机与 GPU runtime 文档

**Files:**
- Create: `docs/docker-training-guide.md`

**Step 1: 记录 Docker 安装步骤**

写明 Ubuntu 22.04 上安装 Docker Engine、Compose plugin 与用户组配置。

**Step 2: 记录 NVIDIA Container Toolkit 安装步骤**

写明 toolkit 仓库配置、安装与 `nvidia-ctk runtime configure` 的执行顺序。

**Step 3: 记录 GPU 自检命令**

列出 `nvidia-smi`、`docker run --gpus all ... nvidia-smi` 等最小验证流程。

### Task 2: 镜像构建文件

**Files:**
- Create: `docker/Dockerfile`
- Create: `.dockerignore`

**Step 1: 选择基础镜像**

使用 `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`，确保容器内有 `nvcc`。

**Step 2: 安装 Miniforge 与 Python 3.11 环境**

在镜像中创建固定 conda 环境，便于训练与推理复用。

**Step 3: 安装 PyTorch、项目依赖和 flash-attn**

按 `torch==2.10.0+cu128`、项目 `requirements` 和 editable install 顺序构建镜像。

### Task 3: 容器使用入口

**Files:**
- Create: `docker/entrypoint.sh`
- Create: `docker/run_train.sh`
- Create: `docker/run_infer.sh`
- Create: `docker-compose.yml`

**Step 1: 编写 entrypoint**

自动激活 conda 环境、导出 CUDA/HF 缓存变量并创建输出目录。

**Step 2: 编写训练脚本**

封装当前 `distributed_training/train_distributed.py` 常用参数与环境变量。

**Step 3: 编写推理脚本**

封装当前 `distributed_training/distributed_inference.py` 常用参数与动态队列相关选项。

**Step 4: 编写 compose 文件**

配置 GPU、共享内存、宿主机目录挂载与持久化缓存目录。

### Task 4: 验证与交付

**Files:**
- Modify: `docs/docker-training-guide.md`

**Step 1: 自查脚本参数与仓库入口一致**

核对训练与推理脚本参数名是否与当前代码一致。

**Step 2: 提供首次落地步骤**

明确从宿主机安装到容器内训练/推理的推荐执行顺序。
