﻿﻿﻿# 受限 Docker 容器构建与镜像固化指南

本文档用于在远程 Ubuntu 22.04 服务器上，以“半数服务器资源上限”的方式，重新创建一个干净的 Docker 工作目录，构建一个带有 `conda`、`CUDA toolkit / nvcc`、`PyTorch cu128`、`flash-attn` 的可复用运行时镜像。后续微调、推理都基于该镜像启动，不再在宿主机直接安装底层依赖。

本文档统一使用以下工作目录：

```bash
/raid5/sh/docker-workspace/flash-attn-safe
```

说明：

- 该目录专门用于安全构建镜像、保存 wheel、缓存和输出
- 业务代码仓库仍挂载自 `/raid5/sh/code/vlm-detect`
- 文档中的 `cat > ... <<'EOF'` 命令会自动创建所需文件，你不需要手动新建“仓库内新增文件”

## 1. 目标与原则

本流程的目标是：

- 删除并清理刚刚用于试验的容器
- 在一个全新目录下准备最小化 Docker 构建上下文
- 构建一个仅包含 `CUDA devel + conda` 的基础镜像
- 启动一个“受限资源”的安装容器，在容器内低并发安装 `flash-attn`
- 将安装成功后的容器固化为镜像，供后续训练和推理直接复用

安全原则：

- 不在宿主机直接编译 `flash-attn`
- 不在 `docker build` 阶段编译 `flash-attn`
- 先构建基础镜像，再在运行中的受限容器内安装 `flash-attn`
- 安装 `flash-attn` 时只使用 `MAX_JOBS=1`
- 先构建 wheel，再安装 wheel，避免污染环境
- 当前仓库不要执行 `pip install -e /workspace`，改用 `PYTHONPATH=/workspace_repo`

## 2. 当前服务器的“半数资源”定义

根据现有 `docker info`，服务器大致资源为：

- CPU: `128`
- 内存: `251.5 GiB`
- GPU: `8`

本文默认的“半数资源上限”定义为：

- CPU 上限: `64`
- 内存上限: `120g`
- 内存 + swap 上限: `128g`
- GPU 可见数量: `4` 张，用于安装验证阶段

说明：

- `flash-attn` 编译本身不需要全部 GPU，因此安装阶段只开放 4 张卡即可
- 后续训练或推理时，可根据需要重新启动容器并开放 `8` 张卡

## 3. 退出并删除当前试验容器

如果你当前仍在容器终端中，先执行：

```bash
exit
```

回到宿主机后，查看现有容器：

```bash
docker ps -a --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}'
```

删除刚刚试验过的相关容器：

```bash
docker rm -f unsloth-base unsloth-finetune fa2-build-safe unsloth-installer unsloth-runtime 2>/dev/null || true
```

如果你不知道容器名，也可以只根据镜像筛选：

```bash
docker ps -a --filter "ancestor=unsloth-base:cu128" --format '{{.ID}} {{.Names}}'
```

如需一并删除旧镜像：

```bash
docker rmi -f unsloth-base:cu128 unsloth-finetune:cu128-fa2 unsloth-safe-base:cu128 unsloth-runtime:cu128-fa2 2>/dev/null || true
```

## 4. 创建一个全新的干净目录

建议把新的 Docker 工作目录放在仓库外，避免把 `output`、`data`、`cache` 混进项目根目录。

```bash
export FLASH_ATTN_SAFE_HOME=/raid5/sh/docker-workspace/flash-attn-safe

mkdir -p ${FLASH_ATTN_SAFE_HOME}/context
mkdir -p ${FLASH_ATTN_SAFE_HOME}/cache/huggingface
mkdir -p ${FLASH_ATTN_SAFE_HOME}/cache/pip
mkdir -p ${FLASH_ATTN_SAFE_HOME}/cache/conda-pkgs
mkdir -p ${FLASH_ATTN_SAFE_HOME}/wheels
mkdir -p ${FLASH_ATTN_SAFE_HOME}/output
mkdir -p ${FLASH_ATTN_SAFE_HOME}/tmp

cd ${FLASH_ATTN_SAFE_HOME}
```

建议约定：

- 代码仓库仍放在 `/raid5/sh/code/vlm-detect`
- 新建的 `${FLASH_ATTN_SAFE_HOME}` 专门用于镜像构建、wheel、缓存、输出

## 5. 生成基础镜像文件

本节所有命令都会自动创建所需文件。你只需要复制执行，不需要手动打开编辑器创建文件。

### 5.1 创建 `.dockerignore`

```bash
cat > ${FLASH_ATTN_SAFE_HOME}/context/.dockerignore <<'EOF'
*.log
*.tmp
*.bak
__pycache__
.git
.idea
.pytest_cache
.ruff_cache
EOF
```

### 5.2 创建 `entrypoint.sh`

```bash
cat > ${FLASH_ATTN_SAFE_HOME}/context/entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

CONDA_DIR="${CONDA_DIR:-/opt/conda}"
CONDA_ENV="${CONDA_ENV:-unsloth-finetune}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

export PATH="${CUDA_HOME}/bin:${CONDA_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/opt/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/opt/cache/pip}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export CUDAARCHS="${CUDAARCHS:-89}"
export MAX_JOBS="${MAX_JOBS:-1}"

mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}" /opt/wheels /opt/output /opt/tmp

if [[ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

exec "$@"
EOF

chmod +x ${FLASH_ATTN_SAFE_HOME}/context/entrypoint.sh
```

### 5.3 创建 `Dockerfile`

注意：这个基础镜像只负责安装 `conda` 和基础系统工具，不在构建阶段安装 `flash-attn`。

```bash
cat > ${FLASH_ATTN_SAFE_HOME}/context/Dockerfile <<'EOF'
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV CONDA_DIR=/opt/conda
ENV CONDA_ENV=unsloth-finetune
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV HF_HOME=/opt/cache/huggingface
ENV PIP_CACHE_DIR=/opt/cache/pip
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST=8.9
ENV CUDAARCHS=89
ENV MAX_JOBS=1

RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    git-lfs \
    jq \
    less \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    openssh-client \
    pkg-config \
    procps \
    tmux \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniforge.sh

SHELL ["/bin/bash", "-lc"]

RUN conda config --system --set auto_activate_base false && \
    conda update -n base -c conda-forge -y conda && \
    conda create -y -n ${CONDA_ENV} python=3.11 && \
    conda clean -afy

COPY entrypoint.sh /usr/local/bin/unsloth-entrypoint.sh
RUN chmod +x /usr/local/bin/unsloth-entrypoint.sh

WORKDIR /workspace_repo

ENTRYPOINT ["/usr/local/bin/unsloth-entrypoint.sh"]
CMD ["/bin/bash"]
EOF
```

## 6. 构建基础镜像

```bash
cd ${FLASH_ATTN_SAFE_HOME}/context
docker build -t unsloth-safe-base:cu128 .
```

构建成功后可查看：

```bash
docker images | grep unsloth-safe-base
```

## 7. 启动“半数资源上限”的受限安装容器

注意：

- 这里不使用 `--rm`，因为后续要把安装好的容器 `commit` 成镜像
- 安装阶段只开放 4 张 GPU，用于验证环境
- CPU / 内存上限设置为半数资源量级

```bash
docker run -it \
  --name unsloth-installer \
  --gpus '"device=0,1,2,3"' \
  --cpus=64 \
  --memory=120g \
  --memory-swap=128g \
  --pids-limit=2048 \
  --ipc=private \
  --shm-size=16g \
  -e CUDA_HOME=/usr/local/cuda \
  -e PATH=/usr/local/cuda/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -e LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  -e MAX_JOBS=1 \
  -e TORCH_CUDA_ARCH_LIST=8.9 \
  -e CUDAARCHS=89 \
  -e HF_HOME=/opt/cache/huggingface \
  -e PIP_CACHE_DIR=/opt/cache/pip \
  -v /raid5/sh/code/vlm-detect:/workspace_repo \
  -v ${FLASH_ATTN_SAFE_HOME}/cache/huggingface:/opt/cache/huggingface \
  -v ${FLASH_ATTN_SAFE_HOME}/cache/pip:/opt/cache/pip \
  -v ${FLASH_ATTN_SAFE_HOME}/cache/conda-pkgs:/opt/conda/pkgs \
  -v ${FLASH_ATTN_SAFE_HOME}/wheels:/opt/wheels \
  -v ${FLASH_ATTN_SAFE_HOME}/output:/opt/output \
  -v ${FLASH_ATTN_SAFE_HOME}/tmp:/opt/tmp \
  unsloth-safe-base:cu128
```

## 8. 容器内安装 conda 环境依赖

以下命令都在 `unsloth-installer` 容器内部执行。

### 8.1 基础检查

```bash
which python
python -V
which nvcc
nvcc --version
echo $CUDA_HOME
nvidia-smi
```

预期：

- `python` 路径位于 `/opt/conda/envs/unsloth-finetune/bin/python`
- `nvcc` 路径位于 `/usr/local/cuda/bin/nvcc`

### 8.2 安装 PyTorch 与项目依赖

不要执行 `pip install -e /workspace_repo`，当前仓库平铺目录较多，editable install 会失败。

```bash
cd /workspace_repo
export PYTHONPATH=/workspace_repo:$PYTHONPATH
export TMPDIR=/opt/tmp

pip install -U pip setuptools wheel packaging ninja
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r /workspace_repo/requirements/distributed-training.txt
pip install pynvml
```

### 8.3 先构建 `flash-attn` wheel，再安装

为避免直接安装过程中失败后污染环境，推荐先构建 wheel：

```bash
cd /workspace_repo
export PYTHONPATH=/workspace_repo:$PYTHONPATH
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDAARCHS=89
export TMPDIR=/opt/tmp

nice -n 10 python -m pip wheel \
  --no-cache-dir \
  --no-build-isolation \
  --no-deps \
  flash-attn==2.8.3 \
  -w /opt/wheels
```

如果 wheel 成功生成，再安装：

```bash
ls -lh /opt/wheels
pip install /opt/wheels/flash_attn-*.whl
```

### 8.4 安装后的验证

```bash
cd /workspace_repo
export PYTHONPATH=/workspace_repo:$PYTHONPATH

python - <<'PY'
import importlib

mods = ["torch", "flash_attn", "xformers", "unsloth", "triton"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(m, "OK", getattr(mod, "__file__", None))
    except Exception as e:
        print(m, "FAIL", repr(e))

import torch
print("torch:", torch.__version__)
print("cuda :", torch.version.cuda)
print("bf16 :", torch.cuda.is_bf16_supported())
print("gpu_count:", torch.cuda.device_count())
PY
```

如果需要更详细的后端报告：

```bash
cd /workspace_repo
export PYTHONPATH=/workspace_repo:$PYTHONPATH
python scripts/check_flash_attention_env.py
cat flash_attention_env_report.json
```

## 9. 将安装成功的容器固化成镜像

当上一步验证通过后，在容器中执行：

```bash
exit
```

回到宿主机后执行：

```bash
docker commit unsloth-installer unsloth-runtime:cu128-fa2
docker images | grep unsloth-runtime
```

如果镜像已生成，可删除安装容器：

```bash
docker rm -f unsloth-installer
```

## 10. 后续如何启动这个镜像

### 10.1 启动一个日常交互容器

这个容器用于日常进入环境、查看包、做小规模验证。

```bash
docker run -it \
  --name unsloth-dev \
  --gpus '"device=0,1,2,3"' \
  --cpus=64 \
  --memory=120g \
  --memory-swap=128g \
  --ipc=private \
  --shm-size=16g \
  -e CUDA_HOME=/usr/local/cuda \
  -e HF_HOME=/opt/cache/huggingface \
  -e PIP_CACHE_DIR=/opt/cache/pip \
  -e PYTHONPATH=/workspace_repo \
  -v /raid5/sh/code/vlm-detect:/workspace_repo \
  -v ${FLASH_ATTN_SAFE_HOME}/cache/huggingface:/opt/cache/huggingface \
  -v ${FLASH_ATTN_SAFE_HOME}/cache/pip:/opt/cache/pip \
  -v ${FLASH_ATTN_SAFE_HOME}/output:/opt/output \
  unsloth-runtime:cu128-fa2
```

### 10.2 退出、停止、重启、进入容器

在容器内退出：

```bash
exit
```

在宿主机停止容器：

```bash
docker stop unsloth-dev
```

重新启动容器：

```bash
docker start unsloth-dev
```

再次进入容器：

```bash
docker exec -it unsloth-dev bash
```

删除容器：

```bash
docker rm -f unsloth-dev
```

## 11. 使用该镜像执行微调任务

建议训练时使用单独容器名，避免和日常开发容器混用。

### 11.1 启动训练容器

如果你要用全部 8 张卡训练，可这样启动：

```bash
docker run -d \
  --name unsloth-train \
  --gpus all \
  --cpus=64 \
  --memory=120g \
  --memory-swap=128g \
  --ipc=host \
  --shm-size=64g \
  -e CUDA_HOME=/usr/local/cuda \
  -e HF_HOME=/opt/cache/huggingface \
  -e PIP_CACHE_DIR=/opt/cache/pip \
  -e PYTHONPATH=/workspace_repo \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 \
  -v /raid5/sh/code/vlm-detect:/workspace_repo \
  -v ${FLASH_ATTN_SAFE_HOME}/cache/huggingface:/opt/cache/huggingface \
  -v ${FLASH_ATTN_SAFE_HOME}/output:/opt/output \
  -w /workspace_repo \
  unsloth-runtime:cu128-fa2 \
  tail -f /dev/null
```

进入训练容器：

```bash
docker exec -it unsloth-train bash
```

### 11.2 在训练容器中运行微调

下面命令在 `unsloth-train` 容器中执行，记得根据实际数据路径调整：

```bash
cd /workspace_repo
export PYTHONPATH=/workspace_repo:$PYTHONPATH
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

torchrun --nproc_per_node=8 scripts/train_distributed.py \
  --model_name google/gemma-3-4b-it \
  --data_path /workspace_repo/data/train.jsonl \
  --output_dir /opt/output/train_run_01 \
  --use_ddp \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --per_device_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --num_epochs 1 \
  --learning_rate 2e-4 \
  --bf16 \
  --tf32 \
  --cpu_threads_per_rank 6 \
  --dataloader_num_workers 6 \
  --dataloader_prefetch_factor 4 \
  --dataloader_pin_memory \
  --dataloader_persistent_workers \
  --image_load_mode lazy \
  --image_width 512 \
  --image_height 512 \
  --gpu_monitor \
  --gpu_log_dir /opt/output/gpu_logs
```

## 12. 使用该镜像执行推理任务

### 12.1 启动推理容器

```bash
docker run -d \
  --name unsloth-infer \
  --gpus all \
  --cpus=64 \
  --memory=120g \
  --memory-swap=128g \
  --ipc=host \
  --shm-size=64g \
  -e CUDA_HOME=/usr/local/cuda \
  -e HF_HOME=/opt/cache/huggingface \
  -e PIP_CACHE_DIR=/opt/cache/pip \
  -e PYTHONPATH=/workspace_repo \
  -v /raid5/sh/code/vlm-detect:/workspace_repo \
  -v ${FLASH_ATTN_SAFE_HOME}/cache/huggingface:/opt/cache/huggingface \
  -v ${FLASH_ATTN_SAFE_HOME}/output:/opt/output \
  -w /workspace_repo \
  unsloth-runtime:cu128-fa2 \
  tail -f /dev/null
```

进入推理容器：

```bash
docker exec -it unsloth-infer bash
```

### 12.2 在推理容器中运行推理

```bash
cd /workspace_repo
export PYTHONPATH=/workspace_repo:$PYTHONPATH

torchrun --nproc_per_node=8 scripts/distributed_inference.py \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --base_model_path google/gemma-3-4b-it \
  --lora_adapter_path /opt/output/train_run_01 \
  --data_path /workspace_repo/data/infer.json \
  --result_dir /opt/output/infer_run \
  --batch_size 4 \
  --scheduler_mode dynamic_queue \
  --partition_strategy round_robin \
  --load_in_4bit
```

## 13. 如果需要更新镜像

后续如果你在容器内追加安装了依赖，并希望保存到镜像：

```bash
docker commit unsloth-dev unsloth-runtime:cu128-fa2-v2
```

查看镜像：

```bash
docker images | grep unsloth-runtime
```

## 14. 常见问题

### 14.1 为什么这里不用 `pip install -e /workspace_repo`

因为当前仓库是平铺目录结构，`setuptools` 会报 “Multiple top-level packages discovered in a flat-layout”。因此当前流程统一使用：

```bash
export PYTHONPATH=/workspace_repo:$PYTHONPATH
```

### 14.2 如果 `flash-attn` 构建再次失败怎么办

优先查看：

```bash
ls -lh ${FLASH_ATTN_SAFE_HOME}/wheels
docker logs unsloth-installer 2>/dev/null | tail -n 200
dmesg -T | grep -i -E "oom|killed process|out of memory" | tail -n 50
```

### 14.3 如果训练时共享内存不足

训练和推理容器建议使用：

- `--ipc=host`
- `--shm-size=64g`

安装容器不需要这么高的共享内存，所以前面使用了更保守的 `--ipc=private --shm-size=16g`。

## 15. 推荐实际执行顺序

建议严格按以下顺序进行：

1. 退出并删除旧容器
2. 创建 `${FLASH_ATTN_SAFE_HOME}` 干净目录
3. 在 `context/` 中生成 `Dockerfile` 与 `entrypoint.sh`
4. 构建 `unsloth-safe-base:cu128`
5. 启动受限的 `unsloth-installer`
6. 在容器内安装 PyTorch、项目依赖、`flash-attn`
7. 运行 `check_flash_attention_env.py` 验证
8. `docker commit` 生成 `unsloth-runtime:cu128-fa2`
9. 基于该镜像分别启动训练容器和推理容器

