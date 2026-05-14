#!/usr/bin/env python
"""Unsloth分布式训练脚本 - Gemma 4-E4B (统一配置版)

支持DDP/device_map/FSDP分布式训练模式, 通过DistributedConfig统一配置。
针对8张NVIDIA A6000 GPU (48GB)优化。

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

import gc
import json
import logging
import os
import time
from pathlib import Path

try:
    from distributed_training.distributed_config import (
        DistributedConfig,
        DistributedMode,
        auto_detect_config,
        create_ddp_config,
        create_device_map_config,
        create_fsdp_config,
    )
except ImportError:
    from distributed_config import (
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

from unsloth import FastVisionModel

logging.basicConfig(level=logging.INFO if int(os.environ.get("LOCAL_RANK", 0)) == 0 else logging.WARNING)
logger = logging.getLogger(__name__)


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

    parser.add_argument("--gpu_monitor", action="store_true", default=True)
    parser.add_argument("--gpu_log_dir", type=str, default="gpu_logs")
    parser.add_argument("--gpu_log_interval", type=int, default=50)

    parser.add_argument("--benchmark", action="store_true", help="运行基准测试并输出对比数据")

    return parser.parse_args()


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
        torch.cuda.set_device(local_rank)

        os.environ["NCCL_P2P_LEVEL"] = os.environ.get("NCCL_P2P_LEVEL", "SYS")
        os.environ["NCCL_IB_DISABLE"] = os.environ.get("NCCL_IB_DISABLE", "1")

        if rank == 0:
            logger.info("分布式训练初始化完成")
            logger.info(f"  Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")

            for i in range(world_size):
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


def load_vision_data(data_path, max_workers=8):
    try:
        from distributed_training.dataset import MultimodalDataset
    except ImportError:
        from dataset import MultimodalDataset

    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    mm_dataset = MultimodalDataset(
        data_path=data_path,
        image_load_mode="lazy",
        max_workers=max_workers,
        show_progress=is_main_process(),
    )

    dataset = mm_dataset.to_conversation_list(
        show_memory_stats=is_main_process(),
    )

    if is_main_process():
        stats = mm_dataset.stats()
        logger.info(f"视觉数据集加载完成: {stats['total_samples']} 条")
        logger.info(f"含图片样本: {stats['samples_with_images']} 条")

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


def main():
    args = parse_args()

    config = build_config_from_args(args)

    local_rank, world_size, is_distributed = setup_distributed()

    effective_lr = config.effective_lr
    effective_batch = config.effective_global_batch

    if is_main_process():
        print(config.summary())

    os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

    model_kwargs = config.get_model_kwargs()

    if is_main_process():
        logger.info("正在加载模型...")
        dm = model_kwargs.get("device_map")
        dm_desc = str(dm) if dm is not None else "None (每进程独立GPU)"
        logger.info(f"device_map: {dm_desc}")

    model, processor = FastVisionModel.from_pretrained(**model_kwargs)

    if config.vision_mode:
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor

    if is_main_process():
        logger.info(f"模型加载完成，参数量: {model.num_parameters() / 1e9:.2f}B")

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

    gc_enabled = getattr(model, 'gradient_checkpointing', False)
    if hasattr(model, 'gradient_checkpointing_enable'):
        gc_enabled = True

    cache_status = getattr(model.config, 'use_cache', None)
    if gc_enabled and cache_status:
        model.config.use_cache = False
        if is_main_process():
            logger.info("梯度检查点兼容性处理: 缓存已禁用 (KV缓存与梯度检查点不兼容)")
    elif gc_enabled and is_main_process():
        logger.info(f"梯度检查点兼容性检查: 缓存状态={cache_status} (已为正确配置)")

    if config.vision_mode:
        dataset = load_vision_data(config.data_path, max_workers=args.max_workers)
    else:
        dataset = load_text_data(config.data_path, tokenizer)

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

    if config.vision_mode:
        from unsloth.trainer import UnslothVisionDataCollator

        trainer_kwargs["processing_class"] = processor.tokenizer
        trainer_kwargs["data_collator"] = UnslothVisionDataCollator(model, processor)
    else:
        trainer_kwargs["processing_class"] = tokenizer
        trainer_kwargs["dataset_text_field"] = "text"

    if config.gpu_monitor:
        try:
            from distributed_training.gpu_monitor import GPUMonitor, GPUMonitorCallback
        except ImportError:
            from gpu_monitor import GPUMonitor, GPUMonitorCallback

        gpu_monitor_inst = GPUMonitor(
            log_dir=config.gpu_log_dir,
            log_interval=config.gpu_log_interval,
            distributed_rank=int(os.environ.get("RANK", 0)),
        )
        callbacks.append(GPUMonitorCallback(gpu_monitor_inst, print_interval=100))

    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    trainer = SFTTrainer(**trainer_kwargs)

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
        processor.save_pretrained(config.output_dir)

        logger.info(f"模型保存完成: {config.output_dir}")

        config_dict = config.to_dict()
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
