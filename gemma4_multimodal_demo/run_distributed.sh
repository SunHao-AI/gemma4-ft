#!/bin/bash
# Unsloth分布式训练启动脚本
# 支持DDP和FSDP两种模式
# ============================================================
# 【重要】请根据您的实际环境修改以下路径配置
# ============================================================

# 方式1: 使用HuggingFace在线模型（推荐）
MODEL_NAME="unsloth/gemma-4-E4B-it-bnb-4bit"
DATA_PATH="./unsloth_train_data.jsonl"
OUTPUT_DIR="./outputs/gemma4_e4b_lora"

# 方式2: 使用本地模型路径（如果已下载）
# MODEL_NAME="/raid5/sh/model/unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
# DATA_PATH="/path/to/your/unsloth_train_data.jsonl"
# OUTPUT_DIR="/path/to/your/outputs/gemma4_e4b_lora"

MAX_SEQ_LENGTH=2048
BATCH_SIZE=2
GRAD_ACCUM=4
LR=2e-4
EPOCHS=1

# GPU数量（根据实际情况修改）
N_GPUS=4

# ==================== DDP单机多卡 ====================
echo "========== DDP训练（单机多卡） =========="

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=${N_GPUS} \
    train_distributed.py \
    --model_name ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_device_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --num_epochs ${EPOCHS} \
    --use_ddp


# ==================== FSDP单机多卡 ====================
echo ""
echo "========== FSDP训练（单机多卡） =========="

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=${N_GPUS} \
    train_distributed.py \
    --model_name ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR}_fsdp \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate ${LR} \
    --num_epochs ${EPOCHS} \
    --use_fsdp


# ==================== 多机多卡 ====================
# 主节点（node_rank=0）
echo ""
echo "========== 多机多卡训练（主节点） =========="

MASTER_ADDR="192.168.1.1"
MASTER_PORT=29500
N_NODES=2
NODE_RANK=0

torchrun \
    --nnodes=${N_NODES} \
    --nproc_per_node=${N_GPUS} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train_distributed.py \
    --model_name ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR}_multinode \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_device_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --num_epochs ${EPOCHS} \
    --use_fsdp


# ==================== Accelerate方式 ====================
echo ""
echo "========== Accelerate训练 =========="
echo "请先运行: accelerate config"
echo "然后执行以下命令:"

accelerate launch train_distributed.py \
    --model_name ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR}_accelerate \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_device_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --num_epochs ${EPOCHS}