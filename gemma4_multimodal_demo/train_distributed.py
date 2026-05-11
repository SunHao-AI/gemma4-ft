#!/usr/bin/env python
"""Unsloth分布式训练脚本 - Gemma 4-E4B (8x A6000优化版)

支持DDP和FSDP分布式训练模式, 针对8张NVIDIA A6000 GPU (48GB)优化。

启动命令示例:
  # DDP 8卡训练 (推荐)
  torchrun --nproc_per_node=8 train_distributed.py --use_ddp --vision_mode ...

  # FSDP 8卡训练
  torchrun --nproc_per_node=8 train_distributed.py --use_fsdp --vision_mode ...

  # 多机多卡
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
      --master_addr="192.168.1.1" --master_port=29500 \
      train_distributed.py --use_fsdp --vision_mode ...

8x A6000优化要点:
  - BF16混合精度 (A6000 Ampere架构原生支持)
  - 每GPU batch_size=4 (QLoRA E4B仅需~10GB, 48GB充足)
  - 梯度累积=2 (有效batch=4*2*8=64)
  - 学习率线性缩放: lr = base_lr * sqrt(world_size)
  - NCCL P2P通信优化 (A6000 NVLink)
  - GPU显存/利用率实时监控
"""

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import argparse
import torch
import torch.distributed as dist

from unsloth import FastVisionModel

logging.basicConfig(level=logging.INFO if int(os.environ.get("LOCAL_RANK", 0)) == 0 else logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Unsloth分布式训练脚本 (8x A6000优化版)")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--device_map", type=str, default=None, help="模型分片策略 (仅用于模型并行, DDP模式下不应设置)")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)

    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--lr_scaling", type=str, default="linear", choices=["none", "linear", "sqrt"], help="多GPU学习率缩放策略: none=不缩放, linear=线性缩放, sqrt=平方根缩放")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.06, help="预热比例, 加载数据集后自动转换为warmup_steps (v5.2弃用warmup_ratio)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adamw_8bit")

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--use_fsdp", action="store_true")

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


def scale_learning_rate(base_lr, world_size, scaling="sqrt"):
    if scaling == "none" or world_size <= 1:
        return base_lr
    elif scaling == "linear":
        return base_lr * world_size
    elif scaling == "sqrt":
        return base_lr * (world_size**0.5)
    return base_lr


def load_vision_data(data_path, max_workers=8):
    from gemma4_multimodal_demo.dataset import MultimodalDataset

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

    local_rank, world_size, is_distributed = setup_distributed()

    effective_lr = scale_learning_rate(args.learning_rate, world_size, args.lr_scaling)

    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps * world_size

    if is_main_process():
        print("=" * 70)
        print("Unsloth分布式训练 - Gemma 4-E4B (8x A6000优化版)")
        print("=" * 70)
        print(f"模型路径: {args.model_name}")
        print(f"数据路径: {args.data_path}")
        print(f"输出目录: {args.output_dir}")
        print(f"分布式模式: {'DDP' if args.use_ddp else 'FSDP' if args.use_fsdp else '单GPU'}")
        print(f"视觉模式: {'启用' if args.vision_mode else '禁用'}")
        print(f"World Size: {world_size}")
        print(f"混合精度: {'BF16' if args.bf16 else 'FP16' if args.fp16 else 'FP32'}")
        print(f"每GPU批次: {args.per_device_batch_size}")
        print(f"梯度累积: {args.gradient_accumulation_steps}")
        print(f"有效全局批次: {effective_batch}")
        print(f"学习率: {args.learning_rate} -> {effective_lr} (缩放策略: {args.lr_scaling})")
        print(f"优化器: {args.optim}")
        print(f"GPU监控: {'启用' if args.gpu_monitor else '禁用'}")
        print("=" * 70)

    os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

    model_kwargs = {
        "model_name": args.model_name,
        "max_seq_length": args.max_seq_length,
        "dtype": None,
    }

    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    # device_map策略:
    #   DDP模式: 不传device_map, 每进程独立加载完整模型到local_rank GPU
    #   单GPU模式 + device_map指定: 传入device_map (仅用于模型并行, 如大模型31B+)
    #   单GPU模式 + device_map=None: 不传device_map, 模型加载到默认GPU
    #   注意: device_map与DDP互斥, DDP模式下必须为None
    if args.use_ddp and args.device_map is not None:
        logger.warning("DDP模式下不应设置device_map, 已自动忽略 (device_map与数据并行互斥)")
        args.device_map = None

    if args.device_map is not None:
        model_kwargs["device_map"] = args.device_map
    elif not is_distributed:
        model_kwargs["device_map"] = {"": 0}

    if args.vision_mode:
        model_kwargs["disable_log_stats"] = True
        model_load_fn = FastVisionModel.from_pretrained
    else:
        model_load_fn = FastVisionModel.from_pretrained

    if is_main_process():
        logger.info("正在加载模型...")

    model, processor = model_load_fn(**model_kwargs)

    if args.vision_mode:
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor

    if is_main_process():
        logger.info(f"模型加载完成，参数量: {model.num_parameters() / 1e9:.2f}B")

    if is_main_process():
        logger.info("正在配置LoRA...")

    peft_kwargs = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
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
        "random_state": args.seed,
        "use_rslora": False,
        "loftq_config": None,
    }

    if args.vision_mode:
        model = FastVisionModel.get_peft_model(model, **peft_kwargs)
    else:
        model = FastVisionModel.get_peft_model(model, **peft_kwargs)

    if is_main_process():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"可训练参数: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")

    if args.vision_mode:
        dataset = load_vision_data(args.data_path, max_workers=args.max_workers)
    else:
        dataset = load_text_data(args.data_path, tokenizer)

    use_bf16 = args.bf16 and torch.cuda.is_bf16_supported()
    use_fp16 = args.fp16 or (not use_bf16 and not args.bf16)

    warmup_steps = max(1, int(len(dataset) * args.num_epochs / effective_batch * args.warmup_ratio))
    if is_main_process():
        logger.info(f"预热: {args.warmup_ratio} ratio -> {warmup_steps} steps")

    training_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": effective_lr,
        "num_train_epochs": args.num_epochs,
        "warmup_steps": warmup_steps,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "optim": args.optim,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "max_seq_length": args.max_seq_length,
        "packing": False,
        "report_to": "none",
    }

    if args.vision_mode:
        training_kwargs["remove_unused_columns"] = False
        training_kwargs["dataset_text_field"] = ""

    if args.use_ddp:
        training_kwargs["ddp_find_unused_parameters"] = False

    if args.use_fsdp:
        fsdp_config_path = Path(__file__).parent / "fsdp_config.json"
        if fsdp_config_path.exists():
            with open(fsdp_config_path, "r") as f:
                training_kwargs["fsdp_config"] = json.load(f)
        else:
            training_kwargs["fsdp_config"] = {
                "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "fsdp_sharding_strategy": "FULL_SHARD",
                "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
                "fsdp_use_orig_params": True,
                "fsdp_sync_module_states": True,
            }

    from trl import SFTTrainer, SFTConfig

    training_args = SFTConfig(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": training_args,
    }

    callbacks = []

    if args.vision_mode:
        from unsloth.trainer import UnslothVisionDataCollator

        trainer_kwargs["processing_class"] = processor.tokenizer
        trainer_kwargs["data_collator"] = UnslothVisionDataCollator(model, processor)
    else:
        trainer_kwargs["processing_class"] = tokenizer
        trainer_kwargs["dataset_text_field"] = "text"

    if args.gpu_monitor:
        from gemma4_multimodal_demo.gpu_monitor import GPUMonitor, GPUMonitorCallback

        gpu_monitor = GPUMonitor(
            log_dir=args.gpu_log_dir,
            log_interval=args.gpu_log_interval,
            distributed_rank=int(os.environ.get("RANK", 0)),
        )
        callbacks.append(GPUMonitorCallback(gpu_monitor, print_interval=100))

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

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)

        logger.info(f"模型保存完成: {args.output_dir}")

        training_result = {
            "model_name": args.model_name,
            "data_path": args.data_path,
            "output_dir": args.output_dir,
            "distributed_mode": "DDP" if args.use_ddp else "FSDP" if args.use_fsdp else "single",
            "world_size": world_size,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "learning_rate_base": args.learning_rate,
            "learning_rate_effective": effective_lr,
            "lr_scaling": args.lr_scaling,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_global_batch_size": effective_batch,
            "num_epochs": args.num_epochs,
            "max_seq_length": args.max_seq_length,
            "bf16": use_bf16,
            "fp16": use_fp16,
            "optim": args.optim,
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
