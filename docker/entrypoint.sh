#!/usr/bin/env bash
set -euo pipefail

CONDA_DIR="${CONDA_DIR:-/opt/conda}"
CONDA_ENV="${CONDA_ENV:-gemma4}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

export PATH="${CUDA_HOME}/bin:${CONDA_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}" /workspace/output

if [[ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

exec "$@"
