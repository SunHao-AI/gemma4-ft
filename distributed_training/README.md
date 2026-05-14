# Gemma 4 Multimodal Demo Project

这是一个独立的演示项目，展示如何使用 **Unsloth** 框架微调 Google Gemma 4-E4B 多模态模型，并进行目标检测推理。

## 项目结构

```
gemma4_multimodal_demo/
├── notebooks/                           # Jupyter Notebook 文件夹
│   ├── 01-data_preparation-labelme_conversion.ipynb  # 数据准备：LabelMe转Unsloth格式
│   ├── 02-model_finetuning.ipynb                     # 模型微调教程
│   ├── 03-object_detection_demo.ipynb                # 目标检测演示
│   └── 04-model_comparison.ipynb                     # 模型比较分析
├── train_distributed.py                 # 分布式训练脚本
├── run_distributed.sh                   # 分布式训练启动脚本
├── fsdp_config.json                     # FSDP 配置文件
├── unsloth_train_data.jsonl             # 示例训练数据
├── requirements.txt                     # Python 依赖列表
├── README.md                            # 项目说明文档
├── MODEL_COMPARISON_GUIDE.md            # 模型比较指南
└── COMPARISON_SYSTEM_SUMMARY.md         # 比较系统总结
```

## 系统要求

| 项目 | 要求 |
|------|------|
| Python | 3.10+ |
| CUDA | 11.8+ 或 12.1+ |
| GPU VRAM (E4B) | ~10 GB |
| GPU VRAM (31B) | ~22 GB |

## 快速开始

### 1. 安装依赖

```bash
# 使用 pip 安装
pip install -r requirements.txt

# 或使用 uv 安装（更快）
uv pip install -r requirements.txt
```

### 2. 配置路径

打开 `notebooks/02-model_finetuning.ipynb` 或 `notebooks/03-object_detection_demo.ipynb`，根据您的实际环境修改以下配置：

- **模型路径**: 默认使用 HuggingFace 在线模型 `unsloth/gemma-4-E4B-it-bnb-4bit`
- **数据路径**: 默认使用项目自带的 `./unsloth_train_data.jsonl`
- **输出路径**: 默认输出到 `./outputs/` 目录

### 3. 运行微调

打开 `finetune_gemma4_e4b.ipynb` 并按顺序执行各个单元格。

### 4. 运行目标检测

首先完成模型微调，然后打开 `object_detection_demo.ipynb` 进行目标检测演示。

## 分布式训练

对于多 GPU 环境，可以使用分布式训练：

```bash
# DDP 单机多卡
torchrun --nproc_per_node=4 train_distributed.py \
    --model_name unsloth/gemma-4-E4B-it-bnb-4bit \
    --data_path ./unsloth_train_data.jsonl \
    --output_dir ./outputs/gemma4_e4b_lora \
    --use_ddp

# FSDP 单机多卡
torchrun --nproc_per_node=4 train_distributed.py \
    --model_name unsloth/gemma-4-E4B-it-bnb-4bit \
    --data_path ./unsloth_train_data.jsonl \
    --output_dir ./outputs/gemma4_e4b_lora_fsdp \
    --use_fsdp
```

或使用脚本：

```bash
bash run_distributed.sh
```

## 数据格式

训练数据采用 JSONL 格式，每行包含一个对话样本：

```json
{
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "请分析这张图像..."}]},
    {"role": "assistant", "content": [{"type": "text", "text": "检测结果..."}]}
  ],
  "images": ["path/to/image.jpg"]
}
```

## 使用自定义数据

如果您有 LabelMe 格式的标注数据，可以使用 `test.py` 脚本将其转换为 Unsloth 格式：

```bash
python test.py
```

注意：需要修改 `test.py` 中的路径配置。

## 常见问题

### Q: 模型下载失败怎么办？
A: 可以手动下载模型到本地，然后修改配置使用本地路径。

### Q: VRAM 不够怎么办？
A: 
- 使用 FSDP 分片训练
- 减小 `per_device_batch_size` 或增加 `gradient_accumulation_steps`
- 使用更小的模型（E4B）

### Q: LoRA adapter 加载失败？
A: 确保已完成微调并保存了 LoRA adapter 到正确路径。

## 参考资源

- [Unsloth 官方文档](https://unsloth.ai/)
- [Gemma 4 模型](https://huggingface.co/google/gemma-4)
- [TRL 库](https://github.com/huggingface/trl)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)

## 许可证

本项目仅供学习和研究使用。Gemma 模型使用受 Google 许可证约束。