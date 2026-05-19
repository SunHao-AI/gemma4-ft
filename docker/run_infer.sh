#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_PATH="${BASE_MODEL_PATH:-google/gemma-3-4b-it}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-/workspace/output/train_run}"
DATA_PATH="${DATA_PATH:-/workspace/data/infer.json}"
RESULT_DIR="${RESULT_DIR:-/workspace/output/infer_run}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
QUEUE_BATCH_SIZE="${QUEUE_BATCH_SIZE:-}"
SCHEDULER_MODE="${SCHEDULER_MODE:-dynamic_queue}"
PARTITION_STRATEGY="${PARTITION_STRATEGY:-round_robin}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${RESULT_DIR}"

declare -a extra_args=()
if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${EXTRA_ARGS})
fi

declare -a cmd=(
    torchrun
    --nproc_per_node="${NPROC_PER_NODE}"
    /workspace/scripts/distributed_inference.py
    --gpu_ids "${GPU_IDS}"
    --base_model_path "${BASE_MODEL_PATH}"
    --lora_adapter_path "${LORA_ADAPTER_PATH}"
    --data_path "${DATA_PATH}"
    --result_dir "${RESULT_DIR}"
    --batch_size "${BATCH_SIZE}"
    --scheduler_mode "${SCHEDULER_MODE}"
    --partition_strategy "${PARTITION_STRATEGY}"
    --load_in_4bit
)

if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
    cmd+=(--max_eval_samples "${MAX_EVAL_SAMPLES}")
fi

if [[ -n "${QUEUE_BATCH_SIZE}" ]]; then
    cmd+=(--queue_batch_size "${QUEUE_BATCH_SIZE}")
fi

cmd+=("${extra_args[@]}")
exec "${cmd[@]}"
