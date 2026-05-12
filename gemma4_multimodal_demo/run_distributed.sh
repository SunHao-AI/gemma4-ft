#!/bin/bash
# Unsloth分布式训练启动脚本 (统一配置版)
# ============================================================
# 支持DDP/device_map/FSDP/自动检测模式
# 通过DistributedConfig统一配置分布式训练参数
# ============================================================

# 【重要】请根据您的实际环境修改以下路径配置

# 方式1: 使用HuggingFace在线模型
MODEL_NAME="unsloth/gemma-4-E4B-it-bnb-4bit"
# 方式2: 使用本地模型路径
# MODEL_NAME="/raid5/sh/model/unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"

DATA_PATH="./unsloth_train_data.jsonl"
OUTPUT_DIR="./outputs/gemma4_e4b_lora"

MAX_SEQ_LENGTH=2048
N_GPUS=8

# ==================== DDP 8卡训练 (推荐, 小模型) ====================
# 优化配置:
#   - 每GPU batch_size=4 (48GB VRAM充足)
#   - gradient_accumulation=2 (有效batch=4*2*8=64)
#   - BF16混合精度 (A6000原生支持)
#   - 学习率线性缩放: base_lr * world_size
#   - GPU监控自动启用

echo "========== DDP 8卡训练 (推荐模式) =========="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=${N_GPUS} \
    train_distributed.py \
    --model_name ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_device_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 4e-5 \
    --lr_scaling linear \
    --num_epochs 1 \
    --warmup_ratio 0.06 \
    --weight_decay 0.01 \
    --optim adamw_8bit \
    --bf16 \
    --vision_mode \
    --use_ddp \
    --gpu_monitor \
    --gpu_log_dir gpu_logs/ddp_8gpu

# ==================== DDP 8卡 + 2倍吞吐 (models_per_gpu=2) ====================
# 小模型(E4B ~10GB)可放入单卡(48GB), 增加batch_size模拟2倍吞吐
# 有效批次: 4*2*8*2=128 (每GPU批次=8, 等效2个模型同时反向传播)

echo ""
echo "========== DDP 8卡 2倍吞吐 (models_per_gpu=2) =========="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=${N_GPUS} \
    train_distributed.py \
    --model_name ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR}_2x \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_device_batch_size 4 \
    --models_per_gpu 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 4e-5 \
    --lr_scaling linear \
    --num_epochs 1 \
    --bf16 \
    --vision_mode \
    --use_ddp \
    --gpu_monitor \
    --gpu_log_dir gpu_logs/ddp_8gpu_2x

# ==================== device_map 2D并行: 8卡分4组 ====================
# 大模型场景: 每组2卡承载1个完整模型 (组内模型并行, 组间数据并行)
# 8卡 → 4组 → 4路数据并行 × 2卡模型并行
# 注意: nproc_per_node=4 (每组1个进程), 不是8

echo ""
echo "========== device_map 2D并行: 8卡分4组 =========="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=4 \
    train_distributed.py \
    --model_name ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR}_devicemap \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_device_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 4e-5 \
    --lr_scaling linear \
    --num_epochs 1 \
    --bf16 \
    --vision_mode \
    --use_ddp \
    --device_map balanced \
    --gpu_groups '[[0,1],[2,3],[4,5],[6,7]]' \
    --gpu_monitor \
    --gpu_log_dir gpu_logs/devicemap_4group

# ==================== FSDP 8卡训练 (大模型31B+) ====================
# FSDP适用于更大模型(31B+), 参数/梯度/优化器全分片
# 配置: batch_size=2, grad_accum=4 (有效batch=2*4*8=64)

echo ""
echo "========== FSDP 8卡训练 =========="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=${N_GPUS} \
    train_distributed.py \
    --model_name ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR}_fsdp \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 4e-5 \
    --lr_scaling linear \
    --num_epochs 1 \
    --warmup_ratio 0.06 \
    --weight_decay 0.01 \
    --optim adamw_8bit \
    --bf16 \
    --vision_mode \
    --use_fsdp \
    --gpu_monitor \
    --gpu_log_dir gpu_logs/fsdp_8gpu

# ==================== 自动检测模式 ====================
# 根据模型显存需求自动选择DDP/device_map/FSDP

echo ""
echo "========== 自动检测模式 (model_vram_gb=10) =========="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=${N_GPUS} \
    train_distributed.py \
    --model_name ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR}_auto \
    --auto_detect \
    --model_vram_gb 10 \
    --per_device_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 4e-5 \
    --bf16 \
    --vision_mode \
    --gpu_monitor \
    --gpu_log_dir gpu_logs/auto_detect

# ==================== 仅使用4卡 (资源受限时) ====================
echo ""
echo "========== DDP 4卡训练 (备用模式) =========="

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    train_distributed.py \
    --model_name ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR}_4gpu \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_device_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 4e-5 \
    --lr_scaling linear \
    --num_epochs 1 \
    --warmup_ratio 0.06 \
    --bf16 \
    --vision_mode \
    --use_ddp \
    --gpu_monitor \
    --gpu_log_dir gpu_logs/ddp_4gpu

# ==================== 使用配置文件启动 ====================
# 先生成配置文件, 再用配置文件启动

echo ""
echo "========== 配置文件启动示例 =========="
echo "1. 生成配置文件 (Python交互式):"
echo "   from gemma4_multimodal_demo import create_ddp_config"
echo "   config = create_ddp_config(gpu_ids=[0,1,2,3,4,5,6,7], models_per_gpu=2)"
echo "   config.to_json('my_config.json')"
echo ""
echo "2. 使用配置文件启动:"
echo "   torchrun --nproc_per_node=8 train_distributed.py --distributed_config my_config.json"

# ==================== 性能对比说明 ====================
echo ""
echo "========== 性能对比说明 =========="
echo "训练完成后, 对比 training_result.json 中的数据:"
echo "  - DDP 8卡结果: outputs/gemma4_e4b_lora/training_result.json"
echo "  - DDP 8卡2倍吞吐: outputs/gemma4_e4b_lora_2x/training_result.json"
echo "  - device_map 4组结果: outputs/gemma4_e4b_lora_devicemap/training_result.json"
echo "  - FSDP 8卡结果: outputs/gemma4_e4b_lora_fsdp/training_result.json"
echo "  - GPU监控日志: gpu_logs/*/gpu_summary_*.json"
echo ""
echo "对比指标:"
echo "  1. 训练速度: samples_per_second"
echo "  2. 显存效率: vram_utilization_pct"
echo "  3. Loss一致性: train_loss"
echo "  4. GPU负载均衡: 各GPU alloc/util差异"
echo "  5. 总并行路数: total_parallel_backward"