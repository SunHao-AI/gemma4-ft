# DistributedConfig - 统一分布式训练配置系统

> 🚀 一套灵活的分布式训练参数系统，支持根据模型规模和硬件资源自动选择最优分布式策略（DDP/device_map/FSDP）

## 目录

- [概述](#概述)
- [安装要求](#安装要求)
- [快速开始](#快速开始)
- [核心概念](#核心概念)
- [配置详解](#配置详解)
- [使用场景](#使用场景)
- [命令行参数](#命令行参数)
- [配置文件方式](#配置文件方式)
- [Notebook集成](#notebook集成)
- [自动检测模式](#自动检测模式)
- [常见问题与排查](#常见问题与排查)
- [性能对比指南](#性能对比指南)

---

## 概述

### 设计目标

本系统旨在解决多GPU训练场景中的配置复杂性，提供：

| 功能需求 | 实现方式 |
|---------|---------|
| **小模型每卡多模型** | `models_per_gpu` 参数，映射到batch_size缩放 |
| **GPU选择机制** | `gpu_ids` 参数，指定参与训练的GPU列表 |
| **大模型GPU分组** | `gpu_groups` 参数，每组承载1个完整模型 |
| **device_map均衡** | `device_map_strategy` 参数，支持balanced/auto等模式 |
| **统一接口** | `DistributedConfig` 类，一套配置切换所有模式 |

### 分布式模式对比

| 模式 | 适用场景 | 每GPU显存 | 通信开销 | 加速比 |
|-----|---------|----------|---------|-------|
| **DDP** | 小模型（单卡可容纳） | 模型完整副本 | 低（仅梯度all-reduce） | ~N×加速 |
| **device_map** | 大模型（需多卡容纳） | 模型分片 | 中（激活跨卡传输） | 组间数据并行 |
| **FSDP** | 极大模型（31B+） | 参数/梯度/优化器全分片 | 高（参数all-gather） | 显存节省优先 |

### 核心约束

⚠️ **关键规则：`device_map` 与 DDP 互斥**

- 设置 `device_map` → 模型并行（一个模型拆分到多卡，单进程）
- 使用 DDP → 数据并行（每卡完整模型副本，多进程）
- **两者不能同时启用**

---

## 安装要求

### 系统要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (推荐 CUDA 12.1)
- NCCL 通信库

### 依赖安装

```bash
# 核心依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Unsloth框架 (推荐)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes

# HuggingFace生态
pip install transformers datasets
```

### 硬件推荐

| GPU型号 | VRAM | 推荐配置 |
|--------|------|---------|
| NVIDIA A6000 | 48GB | DDP 8卡，batch=4，QLoRA E4B |
| NVIDIA A100 | 40GB/80GB | DDP/FSDP，大模型用FSDP |
| NVIDIA RTX 3090/4090 | 24GB | DDP 4-8卡，小模型 |
| NVIDIA V100 | 16GB/32GB | DDP，batch需调小 |

---

## 快速开始

### 1. 导入配置模块

```python
from gemma4_multimodal_demo import (
    DistributedConfig,
    create_ddp_config,
    create_device_map_config,
    create_fsdp_config,
    auto_detect_config,
)
```

### 2. 创建DDP配置（小模型推荐）

```python
# 8卡DDP训练，每GPU batch=4
config = create_ddp_config(
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    per_device_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=4e-5,
    lr_scaling="linear",
    model_name="unsloth/gemma-4-E4B-it-bnb-4bit",
    data_path="./train_data.jsonl",
    output_dir="./outputs/gemma4_lora",
    vision_mode=True,
)

# 打印配置摘要
print(config.summary())
```

**预期输出：**

```
======================================================================
分布式训练配置摘要
======================================================================
模式: ddp
数据并行组数: 8
每组GPU数: 1
每GPU模型倍数(models_per_gpu): 1
总并行反向传播路数: 8

训练参数:
  每GPU批次: 4
  梯度累积: 2
  有效全局批次: 64
  基础学习率: 4e-05
  有效学习率(linear缩放): 0.000320
  混合精度: BF16
  优化器: adamw_8bit
  LoRA: r=16, alpha=16

硬件配置:
  指定GPU: [0, 1, 2, 3, 4, 5, 6, 7]
  device_map: None (DDP每进程独立GPU)

启动命令:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_distributed.py ...
======================================================================
```

### 3. 生成torchrun启动命令

```python
cmd = config.get_torchrun_command("train_distributed.py")
print(cmd)
```

**输出：**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_distributed.py --model_name unsloth/gemma-4-E4B-it-bnb-4bit --data_path ./train_data.jsonl --output_dir ./outputs/gemma4_lora --per_device_batch_size 4 --gradient_accumulation_steps 2 --learning_rate 4e-05 --lr_scaling linear --max_seq_length 2048 --bf16 --vision_mode --load_in_4bit --gpu_monitor
```

### 4. 执行训练

```bash
# 在终端中执行上述命令
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    train_distributed.py \
    --model_name unsloth/gemma-4-E4B-it-bnb-4bit \
    --data_path ./train_data.jsonl \
    --output_dir ./outputs/gemma4_lora \
    --per_device_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 4e-5 \
    --lr_scaling linear \
    --bf16 \
    --vision_mode \
    --use_ddp
```

---

## 核心概念

### models_per_gpu — 每GPU吞吐倍数

**适用场景：** DDP模式，小模型可放入单卡，通过增加batch_size提升吞吐量

**工作原理：**

```
┌─────────────────────────────────────────────────────────┐
│  GPU 0: 模型完整副本                                      │
│    ├── batch_slice_0 (batch_size=4)                      │
│    ├── batch_slice_1 (batch_size=4)  ← models_per_gpu=2  │
│    └── 反向传播: 2×梯度 → 等效2个模型同时训练             │
├─────────────────────────────────────────────────────────┤
│  GPU 1-7: 同上                                           │
│    └── 总并行路数: 8 GPU × 2 倍 = 16路反向传播           │
└─────────────────────────────────────────────────────────┘
```

**参数映射：**

| models_per_gpu | 实际每GPU batch | 有效全局batch (8卡) |
|----------------|----------------|-------------------|
| 1 | 4 | 4 × 2 × 8 = 64 |
| 2 | 4 × 2 = 8 | 8 × 2 × 8 = 128 |
| 3 | 4 × 3 = 12 | 12 × 2 × 8 = 192 |

**代码示例：**

```python
# 8卡DDP + 2倍吞吐
config = create_ddp_config(
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    models_per_gpu=2,  # 关键参数
    per_device_batch_size=4,
    gradient_accumulation_steps=2,
)

print(f"总并行路数: {config.total_parallel_backward}")  # 输出: 16
print(f"有效全局batch: {config.effective_global_batch}")  # 输出: 128
```

### gpu_groups — GPU分组配置

**适用场景：** device_map模式，大模型需多卡共同容纳

**工作原理：**

```
┌───────────────────────────────────────────────────────────────┐
│  gpu_groups = [[0,1], [2,3], [4,5], [6,7]]                     │
│                                                                │
│  组0 [GPU0, GPU1]: 模型A (device_map="balanced"分片)           │
│    ├── GPU0: layers 0-10                                       │
│    └── GPU1: layers 11-20                                      │
│                                                                │
│  组1 [GPU2, GPU3]: 模型B (独立副本)                             │
│  组2 [GPU4, GPU5]: 模型C                                       │
│  组3 [GPU6, GPU7]: 模型D                                       │
│                                                                │
│  组间数据并行: 4组 × 1进程 = 4路数据并行                         │
│  组内模型并行: 2卡承载1个模型                                    │
└───────────────────────────────────────────────────────────────┘
```

**torchrun进程数：** `nproc_per_node = len(gpu_groups) = 4`（不是8！）

**代码示例：**

```python
# 8卡分4组，每组2卡承载1个大模型
config = create_device_map_config(
    gpu_groups=[[0, 1], [2, 3], [4, 5], [6, 7]],
    device_map_strategy="balanced",
    per_device_batch_size=4,
)

print(f"数据并行组数: {config.num_data_parallel_groups}")  # 输出: 4
print(f"每组GPU数: {config.gpus_per_model}")  # 输出: 2
```

**启动命令：**

```bash
# 注意: nproc_per_node=4, 不是8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=4 \
    train_distributed.py \
    --use_ddp \
    --device_map balanced \
    --gpu_groups '[[0,1],[2,3],[4,5],[6,7]]' \
    --vision_mode
```

### 学习率缩放策略

**原理：** 当有效batch增大时，需相应增大学习率以保持收敛稳定性

| 策略 | 公式 | 适用场景 |
|-----|------|---------|
| `none` | lr = base_lr | 单GPU，batch不变 |
| `linear` | lr = base_lr × world_size | DDP/FSDP标准做法 |
| `sqrt` | lr = base_lr × √world_size | 某些实验性场景 |

**代码示例：**

```python
config = DistributedConfig(
    mode="ddp",
    learning_rate=4e-5,
    lr_scaling="linear",
)

# 8卡DDP
print(f"基础LR: {config.learning_rate}")  # 4e-5
print(f"有效LR: {config.effective_lr}")  # 4e-5 × 8 = 3.2e-4
```

---

## 配置详解

### DistributedConfig 完整参数

```python
DistributedConfig(
    # === 分布式模式 ===
    mode="ddp",  # "ddp" | "device_map" | "fsdp" | "single_gpu"

    # === 小模型DDP参数 ===
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],  # 参与训练的GPU列表
    models_per_gpu=1,  # 每GPU吞吐倍数

    # === 大模型device_map参数 ===
    gpu_groups=[[0, 1], [2, 3]],  # GPU分组
    device_map_strategy="balanced",  # "balanced" | "auto" | "balanced_low_0"
    custom_device_map=None,  # 自定义层分配字典
    max_memory_per_gpu={0: "40GiB", 1: "40GiB"},  # 显存限制

    # === 训练参数 ===
    per_device_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=4e-5,
    lr_scaling="linear",  # "none" | "linear" | "sqrt"
    num_epochs=1,
    warmup_ratio=0.06,
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="adamw_8bit",

    # === LoRA参数 ===
    lora_r=16,
    lora_alpha=16,
    lora_dropout=0,

    # === 精度与量化 ===
    bf16=True,
    fp16=False,
    load_in_4bit=True,
    max_seq_length=2048,

    # === 数据与输出 ===
    model_name="unsloth/gemma-4-E4B-it-bnb-4bit",
    data_path="./train_data.jsonl",
    output_dir="./outputs",

    # === 监控 ===
    gpu_monitor=True,
    gpu_log_dir="gpu_logs",
    gpu_log_interval=50,

    # === 其他 ===
    seed=3407,
    vision_mode=True,
)
```

### 核心方法

| 方法 | 返回值 | 用途 |
|-----|-------|-----|
| `get_device_map()` | dict/str/None | 生成 `from_pretrained` 的 device_map 参数 |
| `get_training_kwargs()` | dict | 生成 SFTConfig 训练参数字典 |
| `get_model_kwargs()` | dict | 生成模型加载参数字典 |
| `get_torchrun_command()` | str | 生成完整 torchrun 启动命令 |
| `get_cuda_visible_devices()` | str | 生成 CUDA_VISIBLE_DEVICES 环境变量 |
| `summary()` | str | 格式化配置摘要，适合打印 |
| `to_dict()/from_dict()` | dict/DistributedConfig | 配置序列化/反序列化 |
| `to_json()/from_json()` | 文件读写 | 配置保存/加载 |

---

## 使用场景

### 场景1: DDP 8卡训练（小模型）

**条件：** 模型可放入单卡（QLoRA E4B ~10GB < A6000 48GB）

```python
config = create_ddp_config(
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    models_per_gpu=1,
    per_device_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=4e-5,
    lr_scaling="linear",
    model_name="unsloth/gemma-4-E4B-it-bnb-4bit",
    data_path="./train_data.jsonl",
    output_dir="./outputs/ddp_8gpu",
    vision_mode=True,
)

print(config.summary())
# 有效batch: 64, 有效LR: 3.2e-4
```

### 场景2: DDP 8卡 + 2倍吞吐

**条件：** 模型较小，单卡显存充足，可增加batch提升吞吐

```python
config = create_ddp_config(
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    models_per_gpu=2,  # 吞吐倍增
    per_device_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=4e-5,
    lr_scaling="linear",
    model_name="unsloth/gemma-4-E4B-it-bnb-4bit",
    output_dir="./outputs/ddp_8gpu_2x",
    vision_mode=True,
)

print(f"总并行路数: {config.total_parallel_backward}")  # 16
print(f"有效batch: {config.effective_global_batch}")  # 128
```

### 场景3: device_map 2D并行（大模型）

**条件：** 模型较大，需多卡共同容纳，但有多个GPU分组

```python
# 8卡分4组，每组2卡承载1个模型
config = create_device_map_config(
    gpu_groups=[[0, 1], [2, 3], [4, 5], [6, 7]],
    device_map_strategy="balanced",
    per_device_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=4e-5,
    lr_scaling="linear",
    model_name="meta-llama/Llama-2-70b-hf",
    output_dir="./outputs/devicemap_4group",
    vision_mode=False,
)

print(f"数据并行组: {config.num_data_parallel_groups}")  # 4
print(f"每组GPU数: {config.gpus_per_model}")  # 2
```

### 场景4: device_map 自定义分组

**条件：** 需精确控制GPU分组方式

```python
# 方案1: 8卡分2组，每组4卡
config = create_device_map_config(
    gpu_groups=[[0, 1, 2, 3], [4, 5, 6, 7]],
    device_map_strategy="balanced",
)

# 方案2: 使用部分GPU
config = create_device_map_config(
    gpu_groups=[[0, 1], [2, 3]],  # 仅用4卡
    device_map_strategy="balanced",
)

# 方案3: 自定义显存限制
config = create_device_map_config(
    gpu_groups=[[0, 1]],
    max_memory_per_gpu={0: "30GiB", 1: "30GiB"},
)
```

### 场景5: FSDP全分片（极大模型）

**条件：** 模型极大（31B+），需全部GPU分片

```python
config = create_fsdp_config(
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    per_device_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=4e-5,
    lr_scaling="linear",
    model_name="meta-llama/Llama-2-70b-hf",
    output_dir="./outputs/fsdp_8gpu",
    fsdp_config={
        "fsdp_sharding_strategy": "FULL_SHARD",
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
    },
)

# FSDP自动加载fsdp_config.json
```

### 场景6: 仅使用部分GPU

```python
# 仅使用GPU 0-3（4卡训练）
config = create_ddp_config(
    gpu_ids=[0, 1, 2, 3],
    per_device_batch_size=4,
    gradient_accumulation_steps=4,  # 补偿batch
    learning_rate=4e-5,
    lr_scaling="linear",
)

# 或使用run_distributed.sh中的fallback模式
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ...
```

---

## 命令行参数

### train_distributed.py 参数列表

```bash
python train_distributed.py --help
```

**核心参数：**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|-----|
| `--model_name` | str | 必填 | 模型路径或HuggingFace ID |
| `--data_path` | str | 必填 | 训练数据JSONL路径 |
| `--output_dir` | str | 必填 | 输出目录 |
| `--use_ddp` | flag | False | 启用DDP模式 |
| `--use_fsdp` | flag | False | 启用FSDP模式 |
| `--models_per_gpu` | int | 1 | 每GPU吞吐倍数 |
| `--gpu_ids` | str | None | GPU列表，如 "0,1,2,3" |
| `--gpu_groups` | str | None | GPU分组JSON，如 "[[0,1],[2,3]]" |
| `--device_map` | str | None | 分片策略：balanced/auto |
| `--distributed_config` | str | None | 配置文件JSON路径 |
| `--auto_detect` | flag | False | 自动检测最优模式 |
| `--model_vram_gb` | float | 10.0 | 模型显存需求(GB)，用于auto_detect |

**训练参数：**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|-----|
| `--per_device_batch_size` | int | 4 | 每GPU批次大小 |
| `--gradient_accumulation_steps` | int | 2 | 梯度累积步数 |
| `--learning_rate` | float | 4e-5 | 基础学习率 |
| `--lr_scaling` | str | linear | LR缩放：none/linear/sqrt |
| `--num_epochs` | int | 1 | 训练轮数 |
| `--warmup_ratio` | float | 0.06 | 预热比例 |
| `--max_seq_length` | int | 2048 | 最大序列长度 |
| `--lora_r` | int | 16 | LoRA秩 |
| `--lora_alpha` | int | 16 | LoRA alpha |
| `--bf16` | flag | True | BF16混合精度 |
| `--vision_mode` | flag | True | 视觉模型模式 |

### 命令行示例

**DDP 8卡：**
```bash
torchrun --nproc_per_node=8 train_distributed.py \
    --model_name unsloth/gemma-4-E4B-it-bnb-4bit \
    --data_path ./train_data.jsonl \
    --output_dir ./outputs/ddp \
    --per_device_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 4e-5 \
    --lr_scaling linear \
    --use_ddp \
    --vision_mode
```

**DDP 8卡 + 2倍吞吐：**
```bash
torchrun --nproc_per_node=8 train_distributed.py \
    --models_per_gpu 2 \
    --use_ddp \
    --vision_mode \
    ...
```

**device_map 2D并行：**
```bash
torchrun --nproc_per_node=4 train_distributed.py \
    --use_ddp \
    --device_map balanced \
    --gpu_groups '[[0,1],[2,3],[4,5],[6,7]]' \
    --vision_mode \
    ...
```

**FSDP 8卡：**
```bash
torchrun --nproc_per_node=8 train_distributed.py \
    --use_fsdp \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --vision_mode \
    ...
```

**自动检测：**
```bash
torchrun --nproc_per_node=8 train_distributed.py \
    --auto_detect \
    --model_vram_gb 10 \
    --vision_mode \
    ...
```

---

## 配置文件方式

### 生成配置文件

```python
from gemma4_multimodal_demo import create_ddp_config

# 创建配置
config = create_ddp_config(
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    models_per_gpu=2,
    per_device_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=4e-5,
    lr_scaling="linear",
    model_name="unsloth/gemma-4-E4B-it-bnb-4bit",
    data_path="./train_data.jsonl",
    output_dir="./outputs/ddp_2x",
    vision_mode=True,
)

# 保存到JSON
config.to_json("my_distributed_config.json")
```

**生成的JSON文件内容：**

```json
{
  "mode": "ddp",
  "gpu_ids": [0, 1, 2, 3, 4, 5, 6, 7],
  "models_per_gpu": 2,
  "per_device_batch_size": 4,
  "gradient_accumulation_steps": 2,
  "learning_rate": 4e-5,
  "lr_scaling": "linear",
  "effective_global_batch": 128,
  "effective_lr": 0.00032,
  "total_parallel_backward": 16,
  "num_data_parallel_groups": 8,
  "gpus_per_model": 1,
  "bf16": true,
  "vision_mode": true,
  "load_in_4bit": true,
  "lora_r": 16,
  "lora_alpha": 16,
  "model_name": "unsloth/gemma-4-E4B-it-bnb-4bit",
  "data_path": "./train_data.jsonl",
  "output_dir": "./outputs/ddp_2x"
}
```

### 使用配置文件启动

```bash
# 方式1: 直接指定配置文件
torchrun --nproc_per_node=8 train_distributed.py \
    --distributed_config my_distributed_config.json

# 方式2: 加载并修改
python -c "
from gemma4_multimodal_demo import DistributedConfig
config = DistributedConfig.from_json('my_config.json')
config.learning_rate = 2e-5
config.to_json('modified_config.json')
"
```

---

## Notebook集成

### 在Jupyter中使用DistributedConfig

⚠️ **注意：** Notebook是单进程环境，无法实现真正的DDP。配置会自动退化为单GPU模式。

```python
# 导入
from gemma4_multimodal_demo import DistributedConfig

# 创建配置（Notebook会自动退化为单GPU）
dist_config = DistributedConfig(
    mode="ddp",
    per_device_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=4e-5,
    vision_mode=True,
    model_name="unsloth/gemma-4-E4B-it-bnb-4bit",
)

# 打印配置（会提示使用torchrun）
print(dist_config.summary())

# 获取device_map（Notebook模式下返回 {"": 0}）
device_map = dist_config.get_device_map()
print(f"device_map: {device_map}")  # {"": 0}

# 加载模型
model, processor = FastVisionModel.from_pretrained(
    model_name=BASE_MODEL_PATH,
    max_seq_length=2048,
    load_in_4bit=True,
    device_map=device_map,  # Notebook: {"": 0}, DDP: None
)

# 获取训练参数
training_kwargs = dist_config.get_training_kwargs()
# Notebook模式下LR不缩放，batch不乘GPU数

# 获取DDP启动命令（提示用户）
ddp_cmd = dist_config.get_torchrun_command("train_distributed.py")
print(f"💡 8卡加速请用: {ddp_cmd}")
```

### Notebook输出示例

```
⚠️ Notebook单进程模式 (检测到8GPU, 但DDP需torchrun)
  模型在GPU 0 (device_map={'': 0})
  batch_size=4, grad_accum=2
  lr=4e-05 (不缩放, DDP才需LR缩放)
  BF16=True
  有效批次=8 (DDP=64)
  💡 8卡加速请用: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_distributed.py ...
```

---

## 自动检测模式

### 原理

根据模型显存需求和GPU资源自动选择最优模式：

```
┌─────────────────────────────────────────────────────────────┐
│  auto_detect_config(model_vram_gb=10)                       │
│                                                             │
│  1. 获取可用GPU显存                                          │
│     min_gpu_vram = 48GB (A6000)                             │
│     usable_vram = 48GB × 0.85 = 40.8GB                      │
│                                                             │
│  2. 判断模式                                                 │
│     model_vram_gb=10 < usable_vram=40.8GB                   │
│     → 模型可放入单卡 → 选择DDP                               │
│                                                             │
│  3. 计算models_per_gpu                                      │
│     models_fit = 40.8GB / 10GB = 4                          │
│     models_per_gpu = min(4, 2) = 2                          │
│                                                             │
│  4. 输出配置                                                 │
│     mode=ddp, models_per_gpu=2                              │
└─────────────────────────────────────────────────────────────┘
```

### 使用方式

```python
from gemma4_multimodal_demo import auto_detect_config

# 自动检测
config = auto_detect_config(
    model_vram_gb=10,  # 模型显存需求
    per_device_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=4e-5,
    model_name="unsloth/gemma-4-E4B-it-bnb-4bit",
    data_path="./train_data.jsonl",
    output_dir="./outputs/auto",
    vision_mode=True,
)

print(config.summary())
```

### 命令行方式

```bash
torchrun --nproc_per_node=8 train_distributed.py \
    --auto_detect \
    --model_vram_gb 10 \
    --model_name unsloth/gemma-4-E4B-it-bnb-4bit \
    --data_path ./train_data.jsonl \
    --output_dir ./outputs/auto \
    --vision_mode
```

### 不同显存需求的结果

| model_vram_gb | 可用显存 | 自动选择模式 | models_per_gpu |
|---------------|---------|-------------|----------------|
| 10GB | 40GB/卡 | DDP | 2 |
| 20GB | 40GB/卡 | DDP | 1 |
| 50GB | 40GB/卡 | device_map (需2卡/组) | 1 |
| 100GB | 40GB/卡 | FSDP (需全卡分片) | 1 |

---

## 常见问题与排查

### 问题1: ValueError - device_map与DDP互斥

**错误信息：**
```
ValueError: DDP模式与device_map互斥: DDP每进程加载完整模型, device_map会触发模型并行导致冲突
```

**原因：** 同时设置了 `--use_ddp` 和 `--device_map`

**解决方案：**
```bash
# DDP模式: 不设置device_map
torchrun ... --use_ddp  # 正确

# device_map模式: 不设置use_ddp，或使用gpu_groups
torchrun --nproc_per_node=4 ... --device_map balanced --gpu_groups '[[0,1],[2,3]]'
```

### 问题2: GPU分组重复

**错误信息：**
```
ValueError: GPU分组中存在重复GPU, 每个GPU只能属于一个分组
```

**原因：** `gpu_groups` 中同一GPU出现在多个分组

**错误示例：**
```python
gpu_groups=[[0, 1], [1, 2]]  # GPU 1重复
```

**正确示例：**
```python
gpu_groups=[[0, 1], [2, 3]]  # 无重复
```

### 问题3: GPU ID超出范围

**错误信息：**
```
ValueError: gpu_ids中7超出范围(0-3), 可用GPU数: 4
```

**原因：** 指定了不存在的GPU

**解决方案：**
```python
# 检查可用GPU数
import torch
print(torch.cuda.device_count())  # 输出: 4

# 调整gpu_ids
gpu_ids=[0, 1, 2, 3]  # 正确
```

### 问题4: Notebook无法实现DDP

**现象：** Notebook中配置8卡DDP，但实际只用1卡

**原因：** Notebook是单进程环境，DDP需要多进程（torchrun）

**解决方案：**
```python
# Notebook中查看提示
config = DistributedConfig(mode="ddp")
print(config.summary())
# 输出会提示: "💡 8卡加速请用: torchrun ..."

# 实际DDP训练需要在终端执行
# 使用get_torchrun_command()获取命令
cmd = config.get_torchrun_command("train_distributed.py")
# 复制命令到终端执行
```

### 问题5: models_per_gpu在device_map模式下无效

**现象：** device_map模式设置 `models_per_gpu=2`，但实际吞吐未倍增

**原因：** device_map模式已将模型分片到多卡，`models_per_gpu` 会自动降级为1

**日志提示：**
```
WARNING: device_map模式下models_per_gpu>1无实际意义(模型已分片到多卡), 已自动设为1
```

**正确做法：**
```python
# device_map模式不需要models_per_gpu
config = create_device_map_config(
    gpu_groups=[[0, 1], [2, 3]],
    device_map_strategy="balanced",
    # models_per_gpu会被忽略
)
```

### 问题6: torchrun进程数错误

**现象：** device_map 2D并行时，设置 `--nproc_per_node=8` 导致错误

**原因：** GPU分组模式下，进程数应为分组数，不是GPU总数

**错误示例：**
```bash
# 8卡分4组，错误地设置nproc=8
torchrun --nproc_per_node=8 ... --gpu_groups '[[0,1],[2,3],[4,5],[6,7]]'
# 错误: 进程数与分组数不匹配
```

**正确示例：**
```bash
# 8卡分4组 → 4个进程
torchrun --nproc_per_node=4 ... --gpu_groups '[[0,1],[2,3],[4,5],[6,7]]'
```

### 问题7: 学习率未缩放

**现象：** DDP训练但学习率未按GPU数缩放，收敛不稳定

**原因：** `lr_scaling="none"` 或 Notebook单进程模式

**解决方案：**
```python
# DDP模式应使用linear缩放
config = DistributedConfig(
    mode="ddp",
    learning_rate=4e-5,
    lr_scaling="linear",  # 正确
)

# Notebook单进程不缩放
# lr_scaling会被自动忽略，lr不乘GPU数
```

---

## 性能对比指南

### 对比不同模式

训练完成后，对比 `training_result.json`：

```json
{
  "distributed_mode": "ddp",
  "world_size": 8,
  "models_per_gpu": 2,
  "total_parallel_backward": 16,
  "learning_rate_base": 4e-5,
  "learning_rate_effective": 0.00032,
  "effective_global_batch_size": 128,
  "train_loss": 0.812,
  "train_runtime_sec": 245.6,
  "samples_per_second": 52.3,
  "peak_vram_gb": 12.4,
  "vram_utilization_pct": 25.8
}
```

### 关键对比指标

| 指标 | 说明 | 优化目标 |
|-----|------|---------|
| `samples_per_second` | 训练吞吐量 | 越高越好 |
| `train_runtime_sec` | 总训练时长 | 越短越好 |
| `peak_vram_gb` | 峰值显存占用 | 不超过GPU容量 |
| `vram_utilization_pct` | 显存利用率 | 70-90%为宜 |
| `train_loss` | 最终Loss | 各模式应相近 |

### GPU监控日志对比

查看 `gpu_logs/*/gpu_summary_*.json`：

```json
{
  "gpu_alloc_gb_avg": 10.2,
  "gpu_alloc_gb_std": 0.3,
  "gpu_util_pct_avg": 85.6,
  "gpu_util_pct_std": 2.1
}
```

**负载均衡判断：**
- `std < 5%` → 负载均衡良好
- `std > 10%` → 负载不均衡，需调整配置

### 示例对比表

| 模式 | 8卡DDP | 8卡DDP+2× | 4组device_map | 8卡FSDP |
|-----|--------|----------|--------------|--------|
| `samples/s` | 48.2 | 85.6 | 32.1 | 28.5 |
| `train_loss` | 0.81 | 0.82 | 0.81 | 0.82 |
| `peak_vram` | 10GB | 18GB | 25GB/卡 | 5GB/卡 |
| `适用模型` | E4B | E4B | 31B+ | 70B+ |

---

## 附录

### 文件结构

```
gemma4_multimodal_demo/
├── distributed_config.py      # 核心配置模块
├── train_distributed.py       # 分布式训练脚本
├── run_distributed.sh         # 启动脚本示例
├── fsdp_config.json           # FSDP配置
├── gpu_monitor.py             # GPU监控
├── __init__.py                # 模块导出
└── notebooks/
    └── 02-model_finetuning.ipynb  # Notebook示例
```

### 参考资源

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [HuggingFace Accelerate](https://huggingface.co/docs/accelerate)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

---

**版本：** v1.0.0
**最后更新：** 2026-05-11
**作者：** gemma4_multimodal_demo team