"""Accelerate ND-Parallel 分布式训练脚本

使用 Hugging Face Accelerate 的 ND-Parallel 功能实现真正的 2D 并行:
- Tensor Parallel (TP): 模型层内分片, 组内 GPU 共同计算
- Data Parallel (DP): 模型副本间梯度同步

与 train_distributed.py 的区别:
- train_distributed.py: 使用 Unsloth + torchrun + device_map
- train_nd_parallel.py: 使用 Accelerate + accelerate launch + ParallelismConfig

启动命令:
    accelerate launch train_nd_parallel.py \
        --config_file accelerate_config.yaml \
        --model_name /path/to/model \
        --data_path /path/to/data.jsonl \
        --output_dir /path/to/output

配置文件示例 (accelerate_config.yaml):
    compute_environment: LOCAL_MACHINE
    distributed_type: ND_PARALLEL
    mixed_precision: bf16
    parallelism_config:
      dp_replicate_size: 4  # 4 个模型副本 (数据并行)
      tp_size: 4            # 每个模型 4 GPU (张量并行)
    num_processes: 16       # 总 GPU 数 = dp × tp
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from accelerate import Accelerator
from accelerate.parallelism_config import ParallelismConfig
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Accelerate ND-Parallel 分布式训练脚本")

    parser.add_argument("--model_name", type=str, required=True, help="模型路径或名称")
    parser.add_argument("--data_path", type=str, required=True, help="训练数据路径 (JSONL)")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")

    parser.add_argument("--max_seq_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="使用 4bit 量化")
    parser.add_argument("--bf16", action="store_true", default=True, help="使用 BF16 混合精度")

    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0, help="LoRA dropout")

    parser.add_argument("--per_device_batch_size", type=int, default=4, help="每设备批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=4e-5, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--warmup_ratio", type=float, default=0.06, help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--optim", type=str, default="adamw_8bit", help="优化器")

    parser.add_argument("--logging_steps", type=int, default=10, help="日志步数")
    parser.add_argument("--save_steps", type=int, default=300, help="保存步数")
    parser.add_argument("--save_total_limit", type=int, default=2, help="保存数量限制")

    parser.add_argument("--seed", type=int, default=3407, help="随机种子")
    parser.add_argument("--vision_mode", action="store_true", default=False, help="视觉模型模式")
    parser.add_argument("--attn_implementation", type=str, default=None, 
                        choices=["sdpa", "flash_attention_2", "eager"], help="注意力实现")

    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallel 组大小 (从 accelerate config 自动获取)")
    parser.add_argument("--dp_shard_size", type=int, default=1, help="FSDP 分片大小")

    parser.add_argument("--dataloader_num_workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--dataloader_pin_memory", action="store_true", default=True)

    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="启用梯度检查点")

    return parser.parse_args()


def setup_accelerator(args):
    """初始化 Accelerator

    Accelerate 会自动从环境变量或 config_file 获取 ParallelismConfig
    """
    mixed_precision = "bf16" if args.bf16 else "no"

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        logger.info(f"Accelerator 初始化完成")
        logger.info(f"混合精度: {mixed_precision}")
        logger.info(f"进程数: {accelerator.num_processes}")
        logger.info(f"本地 rank: {accelerator.local_process_index}")

        if hasattr(accelerator, "parallelism_config"):
            pc = accelerator.parallelism_config
            logger.info(f"ParallelismConfig: dp_replicate={pc.dp_replicate_size}, tp={pc.tp_size}")

    return accelerator


def load_model_and_tokenizer(args, accelerator):
    """加载模型和 tokenizer

    注意: ND-Parallel 模式下，模型加载后会由 Accelerator 自动进行 TP 分片
    """
    if accelerator.is_main_process:
        logger.info(f"正在加载模型: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "pretrained_model_name_or_path": args.model_name,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
    }

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型加载完成，参数量: {total_params / 1e9:.2f}B")

    return model, tokenizer


def setup_lora(model, args, accelerator):
    """配置 LoRA"""
    if accelerator.is_main_process:
        logger.info(f"正在配置 LoRA: r={args.lora_r}, alpha={args.lora_alpha}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    if accelerator.is_main_process:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"可训练参数: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")

    return model


def load_dataset_and_tokenize(args, tokenizer, accelerator):
    """加载并处理数据集"""
    if accelerator.is_main_process:
        logger.info(f"正在加载数据集: {args.data_path}")

    dataset = load_dataset("json", data_files=args.data_path, split="train")

    def tokenize_fn(examples):
        texts = examples.get("text", examples.get("content", examples.get("input", "")))
        if isinstance(texts, str):
            texts = [texts]

        outputs = tokenizer(
            texts,
            max_length=args.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=args.dataloader_num_workers or 4,
    )

    if accelerator.is_main_process:
        logger.info(f"数据集处理完成，样本数: {len(tokenized_dataset)}")

    return tokenized_dataset


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    accelerator = setup_accelerator(args)

    log_level = logging.INFO if accelerator.is_main_process else logging.WARNING
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录已创建: {args.output_dir}")

    accelerator.wait_for_everyone()

    model, tokenizer = load_model_and_tokenizer(args, accelerator)
    model = setup_lora(model, args, accelerator)

    train_dataset = load_dataset_and_tokenize(args, tokenizer, accelerator)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=args.dataloader_pin_memory,
        remove_unused_columns=False,
    )

    if args.dataloader_num_workers is not None:
        training_args.dataloader_num_workers = args.dataloader_num_workers

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    if accelerator.is_main_process:
        logger.info("=" * 70)
        logger.info("ND-Parallel 分布式训练配置摘要")
        logger.info("=" * 70)
        logger.info(f"模型: {args.model_name}")
        logger.info(f"数据: {args.data_path}")
        logger.info(f"输出: {args.output_dir}")
        logger.info(f"进程数: {accelerator.num_processes}")
        logger.info(f"TP 组大小: {args.tp_size}")
        logger.info(f"批次: {args.per_device_batch_size} × {args.gradient_accumulation_steps}")
        logger.info(f"学习率: {args.learning_rate}")
        logger.info(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        logger.info("=" * 70)

    trainer.train()

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("训练完成，正在保存 LoRA adapter...")
        
        unwrapped_model = accelerator.unwrap_model(model)
        
        if hasattr(unwrapped_model, 'save_pretrained'):
            unwrapped_model.save_pretrained(args.output_dir)
        else:
            trainer.save_model(args.output_dir)
        
        tokenizer.save_pretrained(args.output_dir)
        
        adapter_config_path = Path(args.output_dir) / "adapter_config.json"
        if adapter_config_path.exists():
            logger.info(f"LoRA adapter 已保存到: {args.output_dir}")
            logger.info(f"adapter_config.json 已确认存在")
        else:
            logger.warning(f"警告: adapter_config.json 未找到，请检查保存是否成功")
            logger.info(f"尝试使用 PEFTModel.save_pretrained 直接保存...")
            if hasattr(unwrapped_model, 'peft_config'):
                unwrapped_model.save_pretrained(args.output_dir)
                logger.info(f"第二次保存完成")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()