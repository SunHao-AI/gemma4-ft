#!/usr/bin/env python
"""Unsloth 分布式多模态微调脚本（统一配置版）

支持 DDP/device_map/FSDP 分布式训练模式，通过 DistributedConfig 统一配置。
训练模型与数据格式遵循 Unsloth 视觉微调接口，可用于 Gemma 4 等视觉语言模型。

启动命令示例:
  # DDP 8卡训练 (推荐, 小模型)
  torchrun --nproc_per_node=8 train_distributed.py --use_ddp --vision_mode ...

  # DDP 8卡 + 2倍吞吐 (小模型, models_per_gpu=2)
  torchrun --nproc_per_node=8 train_distributed.py --use_ddp --models_per_gpu 2 --vision_mode ...

  # device_map 2D并行: 8卡分4组, 每组2卡承载1个大模型
  torchrun --nproc_per_node=4 train_distributed.py --use_ddp \
      --device_map balanced --gpu_groups '[[0,1],[2,3],[4,5],[6,7]]' --vision_mode ...

  # FSDP 8卡训练 (大模型31B+)
  torchrun --nproc_per_node=8 train_distributed.py --use_fsdp --vision_mode ...

  # 多机多卡
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
      --master_addr="192.168.1.1" --master_port=29500 \
      train_distributed.py --use_fsdp --vision_mode ...

  # 使用配置文件启动
  torchrun --nproc_per_node=8 train_distributed.py --distributed_config config.json

8x A6000优化要点:
  - BF16混合精度 (A6000 Ampere架构原生支持)
  - 每GPU batch_size=4 (QLoRA E4B仅需~10GB, 48GB充足)
  - 梯度累积=2 (有效batch=4*2*8=64)
  - 学习率线性缩放: lr = base_lr * world_size
  - NCCL P2P通信优化 (A6000 NVLink)
  - GPU显存/利用率实时监控
"""
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")


def _early_setup_gpu_groups():
    """检测 GPU 组配置并返回映射信息

    注意：由于 NCCL 与进程级 CUDA_VISIBLE_DEVICES 不兼容，
    此函数不再修改 CUDA_VISIBLE_DEVICES，只返回映射信息用于：
    1. 控制 max_memory 配置（限制模型只在特定 GPU 上加载）
    2. 在 setup_distributed() 中使用逻辑 GPU 0

    环境变量:
        LOCAL_RANK: torchrun 设置的进程本地 rank
        GPU_GROUPS_JSON: GPU 分组配置 (JSON 格式)

    使用方式:
        torchrun --nproc_per_node=3 train_distributed.py \
            --gpu_groups '[[0,1],[2,3],[6,7]]' --device_map balanced ...
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    gpu_groups_json = os.environ.get("GPU_GROUPS_JSON")
    if gpu_groups_json is None:
        for i, arg in enumerate(sys.argv):
            if arg == "--gpu_groups" and i + 1 < len(sys.argv):
                gpu_groups_json = sys.argv[i + 1]
                break

    if gpu_groups_json is None:
        return None

    try:
        gpu_groups = json.loads(gpu_groups_json)
    except json.JSONDecodeError:
        return None

    if not isinstance(gpu_groups, list) or len(gpu_groups) == 0:
        return None

    if local_rank >= len(gpu_groups):
        return None

    group = gpu_groups[local_rank]

    return (group, list(range(len(group))))


_EARLY_GPU_MAPPING = _early_setup_gpu_groups()


import unsloth  # noqa: F401

import gc
import logging
import statistics
import time
from pathlib import Path

from unsloth_finetune.training.distributed.distributed_config import (
    DistributedConfig,
    DistributedMode,
    auto_detect_config,
    create_ddp_config,
    create_device_map_config,
    create_fsdp_config,
)

import argparse
import torch
import torch.distributed as dist
from unsloth_finetune.core.runtime import (
    configure_root_logging,
    configure_unsloth_compile_cache,
    get_env_value,
    resolve_notebook_dir,
)

NOTEBOOK_DIR = resolve_notebook_dir(
    cwd=Path.cwd(),
    notebook_file=get_env_value("UNSLOTH_NOTEBOOK_FILE", "GEMMA4_NOTEBOOK_FILE"),
)
UNSLOTH_CACHE_DIR = configure_unsloth_compile_cache(NOTEBOOK_DIR)

from unsloth import FastVisionModel
from transformers import TrainerCallback

configure_root_logging(level=logging.INFO if int(os.environ.get("LOCAL_RANK", 0)) == 0 else logging.WARNING)
logger = logging.getLogger(__name__)

from unsloth_finetune.training.distributed.adapter_utils import normalize_saved_adapter_config


def parse_args():
    parser = argparse.ArgumentParser(description="Unsloth分布式训练脚本 (统一配置版)")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_load_in_4bit", action="store_true", default=False)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)

    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--lr_scaling", type=str, default="linear", choices=["none", "linear", "sqrt"])
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adamw_8bit")

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--use_fsdp", action="store_true")

    parser.add_argument("--models_per_gpu", type=int, default=1, help="每GPU吞吐量倍数 (DDP小模型模式), 映射到batch_size缩放. 例: 8卡×2倍=16路并行")
    parser.add_argument("--gpu_ids", type=str, default=None, help="参与训练的GPU列表, 如'0,1,2,3,4,5,6,7'")
    parser.add_argument("--gpu_groups", type=str, default=None, help="GPU分组配置(JSON), 如'[[0,1],[2,3],[4,5],[6,7]]'. 每组承载1个完整模型(组内模型并行, 组间数据并行)")
    parser.add_argument("--device_map", type=str, default=None, help="模型分片策略: balanced/auto/balanced_low_0 (仅用于模型并行模式, DDP模式下不应设置)")
    parser.add_argument("--max_memory_per_gpu", type=str, default=None, help='每GPU最大可用显存(JSON), 如\'{"0":"40GiB","1":"40GiB"}\'')
    parser.add_argument("--distributed_config", type=str, default=None, help="DistributedConfig JSON配置文件路径 (覆盖其他参数)")

    parser.add_argument("--auto_detect", action="store_true", help="自动检测最优分布式模式 (根据模型显存需求和GPU资源)")
    parser.add_argument("--model_vram_gb", type=float, default=10.0, help="模型所需显存(GB), 用于auto_detect模式")

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--vision_mode", action="store_true", default=True)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--cpu_threads_per_rank", type=int, default=None, help="每个rank允许的PyTorch/OMP CPU线程数, 默认自动计算")
    parser.add_argument("--dataloader_num_workers", type=int, default=None, help="DataLoader worker数量, 默认按CPU核数自动计算")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=None, help="每个worker预取batch数, 默认自动计算")
    parser.add_argument("--dataloader_pin_memory", action=argparse.BooleanOptionalAction, default=True, help="是否启用DataLoader pin_memory")
    parser.add_argument("--dataloader_persistent_workers", action=argparse.BooleanOptionalAction, default=True, help="是否启用persistent_workers")
    parser.add_argument("--dataloader_drop_last", action=argparse.BooleanOptionalAction, default=False, help="是否丢弃最后不完整batch")
    parser.add_argument("--ddp_find_unused_parameters", action=argparse.BooleanOptionalAction, default=False, help="DDP是否查找未使用参数, LoRA场景默认关闭提升性能")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="是否启用TF32矩阵乘与cuDNN加速")
    parser.add_argument("--image_load_mode", type=str, default="lazy", choices=["preload", "lazy", "batch"], help="视觉样本图片加载模式")
    parser.add_argument("--image_batch_size", type=int, default=None, help="batch模式图片预加载批大小")
    parser.add_argument("--materialize_vision_dataset", action=argparse.BooleanOptionalAction, default=False, help="是否在训练前将视觉数据集整体物化为list")
    parser.add_argument("--image_width", type=int, default=512, help="视觉训练输入图片宽度")
    parser.add_argument("--image_height", type=int, default=512, help="视觉训练输入图片高度")
    parser.add_argument(
        "--attn_implementation", type=str, default=None, choices=["sdpa", "flash_attention_2", "eager"], help="注意力实现方式: sdpa(推荐), flash_attention_2, eager. None则由Unsloth自动选择"
    )

    parser.add_argument("--gpu_monitor", action="store_true", default=True)
    parser.add_argument("--gpu_log_dir", type=str, default="gpu_logs")
    parser.add_argument("--gpu_log_interval", type=int, default=50)

    parser.add_argument("--benchmark", action="store_true", help="运行基准测试并输出对比数据")

    return parser.parse_args()


def detect_attention_backends() -> dict:
    """检测FA2/xFormers等注意力后端可用性"""
    info = {
        "flash_attn_importable": False,
        "flash_attn_version": None,
        "flash_attn_error": None,
        "xformers_importable": False,
        "xformers_version": None,
        "xformers_error": None,
        "torch_cuda_flash_sdp_enabled": None,
        "torch_mem_efficient_sdp_enabled": None,
    }

    try:
        import flash_attn  # type: ignore

        info["flash_attn_importable"] = True
        info["flash_attn_version"] = getattr(flash_attn, "__version__", None)
    except Exception as exc:
        info["flash_attn_error"] = str(exc)

    try:
        import xformers  # type: ignore

        info["xformers_importable"] = True
        info["xformers_version"] = getattr(xformers, "__version__", None)
    except Exception as exc:
        info["xformers_error"] = str(exc)

    if hasattr(torch.backends, "cuda"):
        cuda_backend = torch.backends.cuda
        if hasattr(cuda_backend, "flash_sdp_enabled"):
            info["torch_cuda_flash_sdp_enabled"] = cuda_backend.flash_sdp_enabled()
        if hasattr(cuda_backend, "mem_efficient_sdp_enabled"):
            info["torch_mem_efficient_sdp_enabled"] = cuda_backend.mem_efficient_sdp_enabled()

    return info


def auto_tune_runtime(args, world_size: int) -> dict:
    """按当前CPU/GPU资源自动推导数据管线与CPU线程参数"""
    cpu_count = os.cpu_count() or max(1, world_size)
    ranks = max(1, world_size)
    cpu_per_rank = max(4, cpu_count // ranks)

    if args.cpu_threads_per_rank is None:
        args.cpu_threads_per_rank = max(2, min(8, cpu_per_rank // 2))

    remaining_cpu = max(2, cpu_per_rank - args.cpu_threads_per_rank)
    if args.dataloader_num_workers is None:
        if args.vision_mode:
            args.dataloader_num_workers = max(2, min(12, remaining_cpu))
        else:
            args.dataloader_num_workers = max(1, min(8, remaining_cpu // 2))

    if args.dataloader_num_workers == 0:
        args.dataloader_prefetch_factor = None
        args.dataloader_persistent_workers = False
    elif args.dataloader_prefetch_factor is None:
        args.dataloader_prefetch_factor = 4 if args.vision_mode else 2

    # 多模态模型(vision_mode)包含多种子模块(audio_tower, vision_encoder等)
    # 训练数据可能只使用部分模态，导致某些参数未参与loss计算
    # 因此需要启用find_unused_parameters以允许DDP正确处理梯度同步
    if args.vision_mode and not args.ddp_find_unused_parameters:
        args.ddp_find_unused_parameters = True
        logger.info("vision_mode启用: 自动设置 ddp_find_unused_parameters=True (多模态模型可能包含未使用参数)")

    if args.image_load_mode == "batch" and args.image_batch_size is None:
        prefetch = args.dataloader_prefetch_factor or 2
        args.image_batch_size = max(64, args.per_device_batch_size * prefetch * 4)

    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads_per_rank)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads_per_rank)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.cpu_threads_per_rank)

    torch.set_num_threads(args.cpu_threads_per_rank)
    interop_threads = max(1, min(4, args.cpu_threads_per_rank // 2))
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        logger.debug("torch.set_num_interop_threads 已被初始化，跳过重复设置")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    return {
        "cpu_count": cpu_count,
        "world_size": world_size,
        "cpu_per_rank": cpu_per_rank,
        "cpu_threads_per_rank": args.cpu_threads_per_rank,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_prefetch_factor": args.dataloader_prefetch_factor,
        "dataloader_pin_memory": args.dataloader_pin_memory,
        "dataloader_persistent_workers": args.dataloader_persistent_workers,
        "image_load_mode": args.image_load_mode,
        "image_batch_size": args.image_batch_size,
        "materialize_vision_dataset": args.materialize_vision_dataset,
        "tf32": args.tf32,
    }


class TrainingPerformanceCallback(TrainerCallback):
    """记录step时延等轻量性能指标"""

    def __init__(self, output_dir: str, distributed_rank: int = 0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.distributed_rank = distributed_rank
        self.step_durations: list[float] = []
        self._step_start: float | None = None
        self.summary_path = self.output_dir / "performance_summary.json"

    def on_step_begin(self, args, state, control, **kwargs):
        if self.distributed_rank == 0:
            self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self.distributed_rank != 0 or self._step_start is None:
            return
        self.step_durations.append(time.perf_counter() - self._step_start)
        self._step_start = None

    def build_summary(self) -> dict:
        if not self.step_durations:
            return {"steps_observed": 0}

        sorted_steps = sorted(self.step_durations)
        p95_index = min(len(sorted_steps) - 1, max(0, int(len(sorted_steps) * 0.95) - 1))
        tail = self.step_durations[-min(5, len(self.step_durations)) :]
        return {
            "steps_observed": len(self.step_durations),
            "first_step_sec": round(self.step_durations[0], 4),
            "avg_step_sec": round(sum(self.step_durations) / len(self.step_durations), 4),
            "median_step_sec": round(statistics.median(self.step_durations), 4),
            "p95_step_sec": round(sorted_steps[p95_index], 4),
            "steady_state_avg_last5_sec": round(sum(tail) / len(tail), 4),
        }

    def on_train_end(self, args, state, control, **kwargs):
        if self.distributed_rank != 0:
            return
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(self.build_summary(), f, indent=2, ensure_ascii=False)


def build_config_from_args(args) -> DistributedConfig:
    """从命令行参数构建DistributedConfig

    支持三种方式:
      1. --distributed_config: 直接从JSON文件加载
      2. --auto_detect: 根据model_vram_gb自动选择模式
      3. 手动指定: --use_ddp/--use_fsdp + --models_per_gpu/--gpu_groups等
    """
    if args.distributed_config is not None:
        logger.info(f"从配置文件加载: {args.distributed_config}")
        return DistributedConfig.from_json(args.distributed_config)

    if args.auto_detect:
        logger.info(f"自动检测模式, 模型显存需求: {args.model_vram_gb}GB")
        return auto_detect_config(
            model_vram_gb=args.model_vram_gb,
            per_device_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lr_scaling=args.lr_scaling,
            model_name=args.model_name,
            data_path=args.data_path,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit and not args.no_load_in_4bit,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            seed=args.seed,
            bf16=args.bf16,
            vision_mode=args.vision_mode,
            gpu_monitor=args.gpu_monitor,
            gpu_log_dir=args.gpu_log_dir,
            gpu_log_interval=args.gpu_log_interval,
            tf32=args.tf32,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_prefetch_factor=args.dataloader_prefetch_factor,
            dataloader_pin_memory=args.dataloader_pin_memory,
            dataloader_persistent_workers=args.dataloader_persistent_workers,
            dataloader_drop_last=args.dataloader_drop_last,
            ddp_find_unused_parameters=args.ddp_find_unused_parameters,
            cpu_threads_per_rank=args.cpu_threads_per_rank,
            image_load_mode=args.image_load_mode,
            image_batch_size=args.image_batch_size,
            materialize_vision_dataset=args.materialize_vision_dataset,
            attn_implementation=args.attn_implementation,
        )

    gpu_ids = None
    if args.gpu_ids is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]

    gpu_groups = None
    if args.gpu_groups is not None:
        gpu_groups = json.loads(args.gpu_groups)

    max_memory = None
    if args.max_memory_per_gpu is not None:
        max_memory = json.loads(args.max_memory_per_gpu)

    load_4bit = args.load_in_4bit and not args.no_load_in_4bit

    gpu_monitor_kwargs = {
        "gpu_monitor": args.gpu_monitor,
        "gpu_log_dir": args.gpu_log_dir,
        "gpu_log_interval": args.gpu_log_interval,
    }

    if args.use_fsdp:
        return create_fsdp_config(
            gpu_ids=gpu_ids,
            per_device_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lr_scaling=args.lr_scaling,
            model_name=args.model_name,
            data_path=args.data_path,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            load_in_4bit=load_4bit,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            seed=args.seed,
            bf16=args.bf16,
            vision_mode=args.vision_mode,
            **gpu_monitor_kwargs,
            tf32=args.tf32,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_prefetch_factor=args.dataloader_prefetch_factor,
            dataloader_pin_memory=args.dataloader_pin_memory,
            dataloader_persistent_workers=args.dataloader_persistent_workers,
            dataloader_drop_last=args.dataloader_drop_last,
            ddp_find_unused_parameters=args.ddp_find_unused_parameters,
            cpu_threads_per_rank=args.cpu_threads_per_rank,
            image_load_mode=args.image_load_mode,
            image_batch_size=args.image_batch_size,
            materialize_vision_dataset=args.materialize_vision_dataset,
            attn_implementation=args.attn_implementation,
        )

    if args.device_map is not None or gpu_groups is not None:
        return create_device_map_config(
            gpu_groups=gpu_groups,
            device_map_strategy=args.device_map or "balanced",
            per_device_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lr_scaling=args.lr_scaling,
            max_memory_per_gpu=max_memory,
            model_name=args.model_name,
            data_path=args.data_path,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            load_in_4bit=load_4bit,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            seed=args.seed,
            bf16=args.bf16,
            vision_mode=args.vision_mode,
            **gpu_monitor_kwargs,
            tf32=args.tf32,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_prefetch_factor=args.dataloader_prefetch_factor,
            dataloader_pin_memory=args.dataloader_pin_memory,
            dataloader_persistent_workers=args.dataloader_persistent_workers,
            dataloader_drop_last=args.dataloader_drop_last,
            ddp_find_unused_parameters=args.ddp_find_unused_parameters,
            cpu_threads_per_rank=args.cpu_threads_per_rank,
            image_load_mode=args.image_load_mode,
            image_batch_size=args.image_batch_size,
            materialize_vision_dataset=args.materialize_vision_dataset,
            attn_implementation=args.attn_implementation,
        )

    return create_ddp_config(
        gpu_ids=gpu_ids,
        models_per_gpu=args.models_per_gpu,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scaling=args.lr_scaling,
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=load_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        bf16=args.bf16,
        vision_mode=args.vision_mode,
        **gpu_monitor_kwargs,
        tf32=args.tf32,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        dataloader_drop_last=args.dataloader_drop_last,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        cpu_threads_per_rank=args.cpu_threads_per_rank,
        image_load_mode=args.image_load_mode,
        image_batch_size=args.image_batch_size,
        materialize_vision_dataset=args.materialize_vision_dataset,
        attn_implementation=args.attn_implementation,
    )


def setup_distributed():
    if "RANK" not in os.environ:
        logger.info("未检测到分布式环境，使用单GPU模式")
        return 0, 1, False

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl")

        is_gpu_group_mode = _EARLY_GPU_MAPPING is not None

        if is_gpu_group_mode:
            group = _EARLY_GPU_MAPPING[0]
            primary_gpu = group[0]
            torch.cuda.set_device(primary_gpu)
            logger.info(f"GPU组模式: local_rank={local_rank}, 主GPU={primary_gpu}, GPU组={group}")
        else:
            torch.cuda.set_device(local_rank)

        os.environ["NCCL_P2P_LEVEL"] = os.environ.get("NCCL_P2P_LEVEL", "SYS")
        os.environ["NCCL_IB_DISABLE"] = os.environ.get("NCCL_IB_DISABLE", "1")
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

        if rank == 0:
            logger.info("分布式训练初始化完成")
            logger.info(f"  Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
            logger.info(f"  GPU组模式: {is_gpu_group_mode}")

            device_count = torch.cuda.device_count()
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f}GB")

    return local_rank, world_size, is_distributed


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def resolve_image_size(args) -> tuple[int, int]:
    """解析并验证训练图片尺寸参数"""
    if args.image_width <= 0 or args.image_height <= 0:
        raise ValueError(f"图片尺寸必须为正整数, 当前: width={args.image_width}, height={args.image_height}")
    return args.image_width, args.image_height


def load_vision_data(
    data_path,
    max_workers=8,
    image_size: tuple[int, int] | None = None,
    image_load_mode: str = "lazy",
    image_batch_size: int | None = None,
    materialize_dataset: bool = False,
):
    from unsloth_finetune.training.distributed.dataset import MultimodalDataset

    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    mm_dataset = MultimodalDataset(
        data_path=data_path,
        image_load_mode=image_load_mode,
        max_workers=max_workers,
        show_progress=is_main_process(),
        image_size=image_size,
        batch_size=image_batch_size,
    )

    if materialize_dataset:
        dataset = mm_dataset.to_conversation_list(
            show_memory_stats=is_main_process(),
        )
        dataset_kind = "materialized_list"
    else:
        dataset = mm_dataset
        dataset_kind = "lazy_dataset"

    if is_main_process():
        stats = mm_dataset.stats()
        logger.info(f"视觉数据集加载完成: {stats['total_samples']} 条")
        logger.info(f"含图片样本: {stats['samples_with_images']} 条")
        logger.info("数据集供给模式: %s", dataset_kind)
        logger.info(
            "训练图片尺寸: %sx%s",
            image_size[0] if image_size is not None else "原图",
            image_size[1] if image_size is not None else "原图",
        )

    return dataset


def load_text_data(data_path, tokenizer):
    from datasets import load_dataset

    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    dataset = load_dataset("json", data_files=str(data_file), split="train")

    if is_main_process():
        logger.info(f"数据集加载完成: {len(dataset)} 条")

    def format_data(sample):
        messages = sample.get("messages", [])
        if not messages:
            return ""
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    if "messages" in dataset.column_names:
        dataset = dataset.map(
            lambda x: {"text": format_data(x)},
            remove_columns=dataset.column_names,
            desc="格式化数据",
        )

    return dataset


def validate_training_configuration(config, effective_lr: float, world_size: int) -> None:
    """在真正加载模型前校验关键训练超参数，尽早阻断明显异常配置。"""
    if world_size <= 1:
        return

    if config.lr_scaling != "none":
        logger.info(
            "学习率缩放校验: base_lr=%s, effective_lr=%s, strategy=%s, world_size=%s",
            config.learning_rate,
            effective_lr,
            config.lr_scaling,
            world_size,
        )

    # 多 GPU 视觉 LoRA 训练下，过高 effective LR 往往意味着发生了双重缩放。
    if config.vision_mode and effective_lr > 5e-4:
        raise ValueError(
            "effective_lr 过高，疑似发生学习率重复缩放。"
            f"当前 base_lr={config.learning_rate}, effective_lr={effective_lr}, "
            f"world_size={world_size}, lr_scaling={config.lr_scaling}。"
            "请向训练脚本传入未缩放的 base learning rate。"
        )


def setup_gpu_group_visibility(config, local_rank: int) -> tuple[int, int] | None:
    """确认 GPU 组隔离状态并返回映射信息

    注意：CUDA_VISIBLE_DEVICES 已在脚本开头（导入 Unsloth 之前）设置。
    此函数主要用于：
    1. 确认 GPU 组隔离已生效
    2. 将映射信息传递给 DistributedConfig 用于 max_memory 重映射

    Args:
        config: DistributedConfig 实例
        local_rank: 当前进程的 local_rank

    Returns:
        如果配置了 gpu_groups，返回 (原始组内GPU列表, 重映射后的组内GPU列表)
        例如: ([6, 7], [0, 1]) 表示原始 GPU 6,7 被重映射为逻辑 GPU 0,1
        如果未配置 gpu_groups，返回 None
    """
    global _EARLY_GPU_MAPPING

    if config.gpu_groups is None or config.mode != "device_map":
        return None

    if local_rank >= len(config.gpu_groups):
        logger.warning(f"local_rank={local_rank} 超出 gpu_groups 范围，跳过设置")
        return None

    current_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    logger.info(f"GPU组隔离: local_rank={local_rank}, CUDA_VISIBLE_DEVICES={current_cuda_devices}")

    if _EARLY_GPU_MAPPING is not None:
        original_group, remapped_group = _EARLY_GPU_MAPPING
        logger.info(f"GPU映射: 原始={original_group} -> 逻辑={remapped_group}")
        return (original_group, remapped_group)

    group = config.gpu_groups[local_rank]
    remapped = list(range(len(group)))
    return (group, remapped)


def main():
    args = parse_args()
    image_size = resolve_image_size(args)

    local_rank, world_size, is_distributed = setup_distributed()
    runtime_tuning = auto_tune_runtime(args, world_size)
    attention_backends = detect_attention_backends()
    config = build_config_from_args(args)

    gpu_group_mapping = setup_gpu_group_visibility(config, local_rank)

    if gpu_group_mapping is not None:
        config._gpu_group_mapping = gpu_group_mapping

    effective_lr = config.effective_lr
    effective_batch = config.effective_global_batch
    validate_training_configuration(config, effective_lr, world_size)

    if is_main_process():
        print(config.summary())
        logger.info(f"Unsloth compile cache: {UNSLOTH_CACHE_DIR}")
        logger.info("运行时调优: %s", json.dumps(runtime_tuning, ensure_ascii=False))
        logger.info("注意力后端: %s", json.dumps(attention_backends, ensure_ascii=False))

    os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

    model_kwargs = config.get_model_kwargs()
    timings: dict[str, float] = {}

    if is_main_process():
        logger.info("正在加载模型...")
        dm = model_kwargs.get("device_map")
        dm_desc = str(dm) if dm is not None else "None (每进程独立GPU)"
        logger.info(f"device_map: {dm_desc}")

    model_load_start = time.perf_counter()
    model, processor = FastVisionModel.from_pretrained(**model_kwargs)
    timings["model_load_sec"] = round(time.perf_counter() - model_load_start, 4)

    if config.vision_mode:
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor

    if is_main_process():
        logger.info(f"模型加载完成，参数量: {model.num_parameters() / 1e9:.2f}B")
        # 确认注意力实现: 检查模型config的_attn_implementation字段
        _resolved_attn = getattr(model.config, "_attn_implementation", None) or getattr(model.config, "attn_implementation", None)
        logger.info(f"注意力实现: config={_resolved_attn}, requested={config.attn_implementation}")

    if is_main_process():
        logger.info("正在配置LoRA...")

    peft_kwargs = {
        "r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": config.seed,
        "use_rslora": False,
        "loftq_config": None,
    }

    model = FastVisionModel.get_peft_model(model, **peft_kwargs)

    if is_main_process():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"可训练参数: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")

    gc_enabled = getattr(model, "gradient_checkpointing", False)
    if hasattr(model, "gradient_checkpointing_enable"):
        gc_enabled = True

    cache_status = getattr(model.config, "use_cache", None)
    if gc_enabled and cache_status:
        model.config.use_cache = False
        if is_main_process():
            logger.info("梯度检查点兼容性处理: 缓存已禁用 (KV缓存与梯度检查点不兼容)")
    elif gc_enabled and is_main_process():
        logger.info(f"梯度检查点兼容性检查: 缓存状态={cache_status} (已为正确配置)")

    if config.vision_mode:
        data_load_start = time.perf_counter()
        dataset = load_vision_data(
            config.data_path,
            max_workers=args.max_workers,
            image_size=image_size,
            image_load_mode=args.image_load_mode,
            image_batch_size=args.image_batch_size,
            materialize_dataset=args.materialize_vision_dataset,
        )
        timings["dataset_prepare_sec"] = round(time.perf_counter() - data_load_start, 4)
    else:
        data_load_start = time.perf_counter()
        dataset = load_text_data(config.data_path, tokenizer)
        timings["dataset_prepare_sec"] = round(time.perf_counter() - data_load_start, 4)

    training_kwargs = config.get_training_kwargs()

    warmup_steps = max(1, int(len(dataset) * config.num_epochs / config.effective_global_batch * config.warmup_ratio))
    training_kwargs["warmup_steps"] = warmup_steps
    training_kwargs["learning_rate"] = effective_lr
    training_kwargs["output_dir"] = config.output_dir

    if is_main_process():
        logger.info(f"预热: {config.warmup_ratio} ratio -> {warmup_steps} steps")

    if config.mode == DistributedMode.FSDP.value:
        fsdp_cfg = config._load_fsdp_config()
        training_kwargs["fsdp_config"] = fsdp_cfg

    from trl import SFTTrainer, SFTConfig

    training_args = SFTConfig(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": training_args,
    }

    callbacks = []
    performance_callback = TrainingPerformanceCallback(
        output_dir=config.output_dir,
        distributed_rank=int(os.environ.get("RANK", 0)),
    )
    callbacks.append(performance_callback)

    if config.vision_mode:
        from unsloth.trainer import UnslothVisionDataCollator

        trainer_kwargs["processing_class"] = processor.tokenizer
        trainer_kwargs["data_collator"] = UnslothVisionDataCollator(model, processor)
    else:
        trainer_kwargs["processing_class"] = tokenizer
        trainer_kwargs["dataset_text_field"] = "text"

    if config.gpu_monitor:
        from unsloth_finetune.training.distributed.gpu_monitor import GPUMonitor, GPUMonitorCallback

        gpu_monitor_inst = GPUMonitor(
            log_dir=config.gpu_log_dir,
            log_interval=config.gpu_log_interval,
            distributed_rank=int(os.environ.get("RANK", 0)),
        )
        callbacks.append(GPUMonitorCallback(gpu_monitor_inst, print_interval=100))

    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    trainer_init_start = time.perf_counter()
    trainer = SFTTrainer(**trainer_kwargs)
    timings["trainer_init_sec"] = round(time.perf_counter() - trainer_init_start, 4)

    if is_main_process():
        logger.info("开始训练...")
        gpu_stats = torch.cuda.get_device_properties(local_rank)
        start_memory = torch.cuda.max_memory_reserved(local_rank) / 1024**3
        logger.info(f"GPU {local_rank}: {gpu_stats.name}, 初始VRAM: {start_memory:.2f}GB")

    torch.cuda.reset_peak_memory_stats(local_rank)

    start_time = time.time()
    trainer_stats = trainer.train()
    train_time = time.time() - start_time

    metrics = trainer_stats.metrics

    if is_main_process():
        logger.info("训练完成！")

        peak_memory = torch.cuda.max_memory_reserved(local_rank) / 1024**3
        total_gpu_mem = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3

        logger.info(f"峰值VRAM: {peak_memory:.2f}GB / {total_gpu_mem:.1f}GB ({peak_memory / total_gpu_mem * 100:.1f}%)")

        train_loss = metrics.get("train_loss", 0)
        train_runtime = metrics.get("train_runtime", 0)
        samples_per_sec = metrics.get("train_samples_per_second", 0)
        steps_per_sec = metrics.get("train_steps_per_second", 0)

        logger.info(f"训练时长: {train_runtime:.2f}s ({train_time:.2f}s 实际)")
        logger.info(f"最终Loss: {train_loss:.4f}")
        logger.info(f"吞吐量: {samples_per_sec:.2f} samples/s, {steps_per_sec:.4f} steps/s")

        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(config.output_dir)
        normalized_modules = normalize_saved_adapter_config(config.output_dir)
        processor.save_pretrained(config.output_dir)

        logger.info(f"模型保存完成: {config.output_dir}")
        if normalized_modules:
            logger.info(f"LoRA target_modules 已规范化: {len(normalized_modules)} 个模块")

        config_dict = config.to_dict()
        performance_summary = performance_callback.build_summary()
        training_result = {
            "distributed_config": config_dict,
            "distributed_mode": config.mode,
            "world_size": world_size,
            "models_per_gpu": config.models_per_gpu,
            "total_parallel_backward": config.total_parallel_backward,
            "learning_rate_base": config.learning_rate,
            "learning_rate_effective": effective_lr,
            "lr_scaling": config.lr_scaling,
            "per_device_batch_size": config.per_device_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_global_batch_size": effective_batch,
            "num_epochs": config.num_epochs,
            "max_seq_length": config.max_seq_length,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "image_width": image_size[0],
            "image_height": image_size[1],
            "runtime_tuning": runtime_tuning,
            "attention_backends": attention_backends,
            "timings": timings,
            "performance_summary": performance_summary,
            "train_loss": train_loss,
            "train_runtime_sec": train_runtime,
            "samples_per_second": samples_per_sec,
            "steps_per_second": steps_per_sec,
            "peak_vram_gb": round(peak_memory, 2),
            "total_vram_gb": round(total_gpu_mem, 1),
            "vram_utilization_pct": round(peak_memory / total_gpu_mem * 100, 1),
        }

        result_path = output_path / "training_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(training_result, f, indent=2, ensure_ascii=False)

        logger.info(f"训练结果已保存: {result_path}")

    cleanup_distributed()

    if is_main_process():
        print("\n" + "=" * 70)
        print("训练流程全部完成！")
        print("=" * 70)


if __name__ == "__main__":
    main()
