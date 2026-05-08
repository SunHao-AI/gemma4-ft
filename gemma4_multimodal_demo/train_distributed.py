#!/usr/bin/env python
"""
Unsloth分布式训练脚本 - Gemma 4-E4B
支持DDP和FSDP分布式训练模式

启动命令示例：
# DDP单机多卡
torchrun --nproc_per_node=4 train_distributed.py --use_ddp ...

# FSDP单机多卡
torchrun --nproc_per_node=4 train_distributed.py --use_fsdp ...

# 多机多卡
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    --master_addr="192.168.1.1" --master_port=29500 \
    train_distributed.py --use_fsdp ...
"""

import os
import sys
import json
import argparse
import torch
import torch.distributed as dist
from pathlib import Path
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

from unsloth import FastModel


def parse_args():
    parser = argparse.ArgumentParser(description="Unsloth分布式训练脚本")
    
    parser.add_argument("--model_name", type=str, required=True,
                        help="模型路径（本地或HuggingFace ID）")
    parser.add_argument("--data_path", type=str, required=True,
                        help="训练数据路径（JSONL文件）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="最大序列长度")
    parser.add_argument("--load_in_4bit", type=bool, default=True,
                        help="是否使用4-bit量化")
    parser.add_argument("--device_map", type=str, default=None,
                        help="设备映射（balanced/auto等，用于大模型分片）")
    
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA缩放因子")
    parser.add_argument("--lora_dropout", type=float, default=0,
                        help="LoRA dropout率")
    
    parser.add_argument("--per_device_batch_size", type=int, default=2,
                        help="每GPU批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="训练轮数")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪阈值")
    
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存步数")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="最多保存数量")
    
    parser.add_argument("--use_ddp", action="store_true",
                        help="使用DDP分布式训练")
    parser.add_argument("--use_fsdp", action="store_true",
                        help="使用FSDP分布式训练")
    
    parser.add_argument("--seed", type=int, default=3407,
                        help="随机种子")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="使用bfloat16精度")
    parser.add_argument("--fp16", action="store_true",
                        help="使用float16精度")
    
    return parser.parse_args()


def setup_distributed():
    """
    初始化分布式训练环境
    """
    if "RANK" not in os.environ:
        print("未检测到分布式环境，使用单GPU模式")
        return 0, False
    
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    is_distributed = world_size > 1
    
    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        
        if rank == 0:
            print(f"分布式训练初始化完成")
            print(f"  - Rank: {rank}")
            print(f"  - Local Rank: {local_rank}")
            print(f"  - World Size: {world_size}")
    
    return local_rank, is_distributed


def cleanup_distributed():
    """
    清理分布式训练环境
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """
    检查是否为主进程（rank 0）
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def load_and_prepare_data(data_path, tokenizer):
    """
    加载并预处理训练数据
    """
    data_file = Path(data_path)
    
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    dataset = load_dataset("json", data_files=str(data_file), split="train")
    
    if is_main_process():
        print(f"数据集加载完成: {len(dataset)} 条")
    
    def format_data(sample):
        messages = sample.get("messages", [])
        if not messages:
            return ""
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return text
    
    if "messages" in dataset.column_names:
        dataset = dataset.map(
            lambda x: {"text": format_data(x)},
            remove_columns=dataset.column_names,
            desc="格式化数据"
        )
    
    return dataset


def main():
    args = parse_args()
    
    local_rank, is_distributed = setup_distributed()
    
    if is_main_process():
        print("=" * 60)
        print("Unsloth分布式训练 - Gemma 4-E4B")
        print("=" * 60)
        print(f"模型路径: {args.model_name}")
        print(f"数据路径: {args.data_path}")
        print(f"输出目录: {args.output_dir}")
        print(f"分布式模式: {'DDP' if args.use_ddp else 'FSDP' if args.use_fsdp else '单GPU'}")
        print("=" * 60)
    
    model_kwargs = {
        "model_name": args.model_name,
        "max_seq_length": args.max_seq_length,
        "dtype": None,
    }
    
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    
    if is_main_process():
        print("正在加载模型...")
    
    model, tokenizer = FastModel.from_pretrained(**model_kwargs)
    
    if is_main_process():
        print(f"模型加载完成，参数量: {model.num_parameters() / 1e9:.2f}B")
    
    if is_main_process():
        print("正在配置LoRA...")
    
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    
    if is_main_process():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    dataset = load_and_prepare_data(args.data_path, tokenizer)
    
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        bf16=args.bf16 and torch.cuda.is_bf16_supported(),
        fp16=args.fp16 or not torch.cuda.is_bf16_supported(),
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        report_to="none",
        ddp_find_unused_parameters=False if args.use_ddp else None,
        fsdp_config={
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_sharding_strategy": "FULL_SHARD",
        } if args.use_fsdp else None,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    
    if is_main_process():
        print("开始训练...")
        gpu_stats = torch.cuda.get_device_properties(local_rank)
        start_memory = torch.cuda.max_memory_reserved(local_rank) / 1024**3
        print(f"GPU: {gpu_stats.name}")
        print(f"初始VRAM: {start_memory:.2f} GB")
    
    trainer.train()
    
    if is_main_process():
        print("\n训练完成！")
        end_memory = torch.cuda.max_memory_reserved(local_rank) / 1024**3
        print(f"最终VRAM: {end_memory:.2f} GB")
        
        print(f"\n正在保存模型到: {args.output_dir}")
        
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        print("模型保存完成！")
        
        config_path = output_path / "training_config.json"
        config = {
            "model_name": args.model_name,
            "data_path": args.data_path,
            "output_dir": args.output_dir,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "max_seq_length": args.max_seq_length,
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"训练配置已保存到: {config_path}")
    
    cleanup_distributed()
    
    if is_main_process():
        print("\n" + "=" * 60)
        print("训练流程全部完成！")
        print("=" * 60)


if __name__ == "__main__":
    main()