#!/bin/bash
# Unsloth分布式训练启动脚本 (8x A6000优化版)
# ============================================================
# 针对8张NVIDIA A6000 GPU (48GB VRAM)优化配置
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

# ==================== DDP 8卡训练 (推荐) ====================
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

# ==================== FSDP 8卡训练 ====================
# FSDP适用于更大模型(31B+), E4B模型推荐DDP
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

# ==================== 性能对比基准测试 ====================
echo ""
echo "========== 性能对比说明 =========="
echo "训练完成后, 对比 training_result.json 中的数据:"
echo "  - 单GPU结果: outputs/gemma4_e4b_lora/training_result.json"
echo "  - 8GPU结果: outputs/gemma4_e4b_lora/training_result.json"
echo "  - GPU监控日志: gpu_logs/ddp_8gpu/gpu_summary_*.json"
echo ""
echo "对比指标:"
echo "  1. 训练速度: samples_per_second (预期提升6-7x)"
echo "  2. 显存效率: vram_utilization_pct"
echo "  3. Loss一致性: train_loss (应与单GPU相近)"
echo "  4. GPU负载均衡: 各GPU alloc/util差异"