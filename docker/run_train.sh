#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-unsloth/gemma-3-4b-it}"
DATA_PATH="${DATA_PATH:-/workspace/data/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/train_run}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-300}"
IMAGE_WIDTH="${IMAGE_WIDTH:-512}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-512}"
CPU_THREADS_PER_RANK="${CPU_THREADS_PER_RANK:-6}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-6}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
IMAGE_LOAD_MODE="${IMAGE_LOAD_MODE:-lazy}"
GPU_LOG_DIR="${GPU_LOG_DIR:-/workspace/output/gpu_logs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${OUTPUT_DIR}" "${GPU_LOG_DIR}"

declare -a extra_args=()
if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${EXTRA_ARGS})
fi

declare -a cmd=(
    torchrun
    --nproc_per_node="${NPROC_PER_NODE}"
    /workspace/scripts/train_distributed.py
    --model_name "${MODEL_NAME}"
    --data_path "${DATA_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --use_ddp
    --gpu_ids "${GPU_IDS}"
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --num_epochs "${NUM_EPOCHS}"
    --learning_rate "${LEARNING_RATE}"
    --logging_steps "${LOGGING_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --bf16
    --tf32
    --cpu_threads_per_rank "${CPU_THREADS_PER_RANK}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}"
    --dataloader_pin_memory
    --dataloader_persistent_workers
    --image_load_mode "${IMAGE_LOAD_MODE}"
    --image_width "${IMAGE_WIDTH}"
    --image_height "${IMAGE_HEIGHT}"
    --gpu_monitor
    --gpu_log_dir "${GPU_LOG_DIR}"
)

cmd+=("${extra_args[@]}")
exec "${cmd[@]}"
