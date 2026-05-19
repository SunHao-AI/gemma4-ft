FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV CONDA_DIR=/opt/conda
ENV CONDA_ENV=unsloth-finetune
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV HF_HOME=/workspace/.cache/huggingface
ENV PIP_CACHE_DIR=/workspace/.cache/pip
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST=8.9
ENV MAX_JOBS=16

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

RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && \
    conda activate ${CONDA_ENV} && \
    pip install -U pip setuptools wheel packaging ninja && \
    pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

COPY . /opt/project

RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && \
    conda activate ${CONDA_ENV} && \
    pip install -e "/opt/project[finetune,data,monitor,dev]" && \
    pip install flash-attn --no-build-isolation && \
    python - <<'PY'
import importlib
for name in ("torch", "flash_attn", "xformers", "unsloth", "triton"):
    importlib.import_module(name)
print("Base image dependencies verified.")
PY

RUN rm -rf /opt/project/.git /opt/project/.pytest_cache /opt/project/__pycache__

COPY docker/entrypoint.sh /usr/local/bin/unsloth-entrypoint.sh
COPY docker/run_train.sh /usr/local/bin/unsloth-train.sh
COPY docker/run_infer.sh /usr/local/bin/unsloth-infer.sh

RUN chmod +x /usr/local/bin/unsloth-entrypoint.sh /usr/local/bin/unsloth-train.sh /usr/local/bin/unsloth-infer.sh

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/unsloth-entrypoint.sh"]
CMD ["/bin/bash"]
