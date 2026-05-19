# Docker 训练与推理环境指南

本文档提供一套将 Gemma4 微调与推理迁移到 Docker 容器内运行的完整流程。目标是让宿主机只保留 NVIDIA Driver、Docker 和 GPU runtime，`conda`、`CUDA toolkit`、`nvcc`、`flash-attn`、训练依赖和虚拟环境都放到容器内管理。

## 1. 方案说明

当前远程服务器缺少 `nvcc` 与 `CUDA_HOME`，导致 `flash-attn` 无法在宿主机源码安装。容器化方案通过使用 `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` 基础镜像解决这一问题：

- `devel` 镜像自带 `CUDA toolkit` 和 `nvcc`
- 容器内安装 `Miniforge` 并创建 `Python 3.11` 虚拟环境
- 容器内安装 `torch==2.10.0+cu128`、`unsloth`、`xformers`、`flash-attn`
- 训练与推理直接调用当前仓库中的 `distributed_training/train_distributed.py` 与 `distributed_training/distributed_inference.py`

## 2. 宿主机前置条件

宿主机必须满足：

- Ubuntu 22.04 或兼容 Linux 发行版
- NVIDIA 驱动已正常安装
- Docker Engine 已安装
- `nvidia-container-toolkit` 已安装并接入 Docker

先执行以下检查：

```bash
nvidia-smi
docker --version
docker info | grep -i runtime
```

如果 `docker` 还不能使用 GPU，请安装 `nvidia-container-toolkit`。

## 3. 宿主机安装 Docker

如果 Docker 尚未安装，可执行：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable docker
sudo systemctl start docker
```

可选：允许当前用户直接执行 Docker：

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 4. 宿主机安装 NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

验证 GPU 容器是否可用：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 nvidia-smi
```

## 5. 仓库内新增文件

本仓库已新增以下文件：

- `docker/Dockerfile`
- `docker/entrypoint.sh`
- `docker/run_train.sh`
- `docker/run_infer.sh`
- `docker-compose.yml`

## 6. 构建镜像

在仓库根目录执行：

```bash
mkdir -p docker-cache/huggingface docker-cache/pip docker-cache/conda-pkgs output
docker compose build --progress=plain
```

如果只想用原生命令：

```bash
docker build -f docker/Dockerfile -t unsloth-finetune:cu128-fa2 .
```

## 7. 启动交互式容器

```bash
docker compose run --rm unsloth
```

或者：

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  -v $(pwd):/workspace \
  -v $(pwd)/docker-cache/huggingface:/workspace/.cache/huggingface \
  -v $(pwd)/docker-cache/pip:/workspace/.cache/pip \
  -v $(pwd)/docker-cache/conda-pkgs:/opt/conda/pkgs \
  -v $(pwd)/output:/workspace/output \
  unsloth-finetune:cu128-fa2
```

进入容器后，默认会自动激活 `unsloth-finetune` conda 环境。

## 8. 容器内环境验证

进入容器后执行：

```bash
which python
python -V
which nvcc
nvcc --version
echo $CUDA_HOME
nvidia-smi
```

再验证关键 Python 包：

```bash
python - <<'PY'
import importlib
mods = ["torch", "flash_attn", "xformers", "unsloth", "triton"]
for name in mods:
    try:
        module = importlib.import_module(name)
        print(name, "OK", getattr(module, "__file__", None))
    except Exception as exc:
        print(name, "FAIL", repr(exc))

import torch
print("torch:", torch.__version__)
print("cuda :", torch.version.cuda)
print("bf16 :", torch.cuda.is_bf16_supported())
print("gpu_count:", torch.cuda.device_count())
PY
```

如果需要更详细的 attention backend 报告：

```bash
python distributed_training/check_flash_attention_env.py
cat flash_attention_env_report.json
```

## 9. 在容器内运行微调

### 9.1 使用辅助脚本

默认脚本路径：

- `unsloth-train.sh`
- `unsloth-infer.sh`

示例：

```bash
MODEL_NAME=google/gemma-3-4b-it \
DATA_PATH=/workspace/data/train.jsonl \
OUTPUT_DIR=/workspace/output/train_run_01 \
GPU_IDS=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
PER_DEVICE_BATCH_SIZE=8 \
GRADIENT_ACCUMULATION_STEPS=2 \
NUM_EPOCHS=1 \
LEARNING_RATE=2e-4 \
IMAGE_WIDTH=512 \
IMAGE_HEIGHT=512 \
unsloth-train.sh
```

如果需要附加额外参数，可通过 `EXTRA_ARGS` 传入：

```bash
EXTRA_ARGS="--save_total_limit 3 --warmup_ratio 0.03 --benchmark" unsloth-train.sh
```

### 9.2 直接调用训练脚本

```bash
torchrun --nproc_per_node=8 /workspace/distributed_training/train_distributed.py \
  --model_name google/gemma-3-4b-it \
  --data_path /workspace/data/train.jsonl \
  --output_dir /workspace/output/train_run_01 \
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
  --gpu_log_dir /workspace/output/gpu_logs
```

## 10. 在容器内运行推理

### 10.1 使用辅助脚本

```bash
MODEL_PATH=/workspace/output/train_run_01 \
INPUT_PATH=/workspace/data/infer.json \
OUTPUT_PATH=/workspace/output/infer_result.json \
GPU_IDS=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
unsloth-infer.sh
```

### 10.2 直接调用分布式推理脚本

```bash
torchrun --nproc_per_node=8 /workspace/distributed_training/distributed_inference.py \
  --model_path /workspace/output/train_run_01 \
  --input_path /workspace/data/infer.json \
  --output_path /workspace/output/infer_result.json \
  --gpu_ids 0,1,2,3,4,5,6,7
```

## 11. Notebook 使用方式

如需在容器内运行 Notebook，可额外安装 JupyterLab：

```bash
pip install jupyterlab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

启动容器时映射端口：

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  unsloth-finetune:cu128-fa2
```

然后在浏览器中访问宿主机的 `8888` 端口。

## 12. 推荐的目录组织

建议在仓库根目录准备以下目录：

```text
data/                    # 训练/验证/推理数据
output/                  # 模型输出、日志、适配器
docker-cache/
  huggingface/           # 模型与数据缓存
  pip/                   # pip 缓存
  conda-pkgs/            # conda 包缓存
```

## 13. 常见问题排查

### 13.1 容器内看不到 GPU

- 确认宿主机 `nvidia-smi` 正常
- 确认 `docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 nvidia-smi` 正常
- 确认 `nvidia-container-toolkit` 已安装并重启过 Docker

### 13.2 `flash-attn` 构建失败

- 使用 `docker compose build --progress=plain` 获取完整日志
- 优先确认 `nvcc --version` 在容器内可见
- 如果报 GCC/ABI 错误，再根据编译日志微调系统编译器或 `flash-attn` 版本

### 13.3 DataLoader 或 `torchrun` 报共享内存不足

- 使用 `--ipc=host`
- 或提高 `--shm-size=64g`

### 13.4 模型和数据每次都重新下载

- 确认 `docker-cache/huggingface` 已正确挂载到 `/workspace/.cache/huggingface`

## 14. 建议的首次落地流程

推荐按以下顺序执行：

1. 在宿主机安装 Docker 与 `nvidia-container-toolkit`
2. 运行 GPU 容器自检命令，确认 Docker 能识别 GPU
3. 在仓库根目录执行 `docker compose build --progress=plain`
4. 运行 `docker compose run --rm gemma4`
5. 在容器中执行 `nvcc --version` 与 Python 依赖导入检查
6. 运行 `python distributed_training/check_flash_attention_env.py`
7. 先进行一轮短程训练验证，再正式启动全量训练


