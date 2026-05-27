"""统一分布式训练配置模块

提供灵活的分布式训练参数系统, 支持根据模型规模和硬件资源选择:
  - DDP数据并行: 小模型, 每卡完整模型, 支持models_per_gpu吞吐量倍增
  - device_map模型并行: 大模型, 模型参数在GPU间均衡分配, 支持GPU分组2D并行
  - FSDP分片并行: 大模型, 参数/梯度/优化器全分片

设计参考:
  - Unsloth框架的FastVisionModel.from_pretrained() device_map机制
  - HuggingFace Accelerate的dispatch_model/infer_auto_device_map
  - PyTorch DDP/FSDP标准分布式模式
  - DeepSpeed ZeRO分片策略

核心概念:
  - models_per_gpu: 小模型吞吐量倍增因子, 映射到batch_size或grad_accum缩放
    例: 8卡 × 2 models_per_gpu = 16路并行反向传播
  - gpu_groups: 大模型GPU分组, 每组承载1个完整模型(组内模型并行, 组间数据并行)
    例: [[0,1], [2,3], [4,5], [6,7]] → 4组×2卡 = 4路数据并行 × 2卡模型并行
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


class DistributedMode(Enum):
    DDP = "ddp"
    DEVICE_MAP = "device_map"
    FSDP = "fsdp"
    SINGLE_GPU = "single_gpu"


class LRScalingStrategy(Enum):
    NONE = "none"
    LINEAR = "linear"
    SQRT = "sqrt"


class DeviceMapStrategy(Enum):
    BALANCED = "balanced"
    AUTO = "auto"
    BALANCED_LOW_0 = "balanced_low_0"
    CUSTOM = "custom"


@dataclass
class DistributedConfig:
    """统一分布式训练配置

    通过简单参数切换DDP/device_map/FSDP模式, 自动计算:
      - 有效批次大小
      - 学习率缩放
      - device_map映射
      - GPU分组分配
      - torchrun启动命令

    使用示例:
      # 1. 小模型DDP: 8卡, 每卡2倍吞吐
      config = DistributedConfig(
          mode="ddp",
          gpu_ids=[0,1,2,3,4,5,6,7],
          models_per_gpu=2,
          per_device_batch_size=2,
          learning_rate=4e-5,
      )

      # 2. 大模型device_map: 8卡分4组, 每组2卡承载1个模型
      config = DistributedConfig(
          mode="device_map",
          gpu_groups=[[0,1], [2,3], [4,5], [6,7]],
          device_map_strategy="balanced",
          per_device_batch_size=4,
          learning_rate=4e-5,
      )

      # 3. 大模型FSDP: 8卡全分片
      config = DistributedConfig(
          mode="fsdp",
          gpu_ids=[0,1,2,3,4,5,6,7],
          per_device_batch_size=2,
          learning_rate=4e-5,
      )
    """

    mode: str = "ddp"

    gpu_ids: Optional[List[int]] = None

    models_per_gpu: int = 1

    gpu_groups: Optional[List[List[int]]] = None

    device_map_strategy: Optional[str] = None

    custom_device_map: Optional[Dict[str, Any]] = None

    max_memory_per_gpu: Optional[Dict[int, str]] = None

    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 4e-5
    lr_scaling: str = "linear"
    num_epochs: int = 1
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optim: str = "adamw_8bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0
    seed: int = 3407
    bf16: bool = True
    fp16: bool = False
    vision_mode: bool = True
    tf32: bool = True
    dataloader_num_workers: Optional[int] = None
    dataloader_prefetch_factor: Optional[int] = None
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_drop_last: bool = False
    ddp_find_unused_parameters: bool = False
    cpu_threads_per_rank: Optional[int] = None
    image_load_mode: str = "lazy"
    image_batch_size: Optional[int] = None
    materialize_vision_dataset: bool = False
    attn_implementation: Optional[str] = None

    logging_steps: int = 10
    save_steps: int = 300
    save_total_limit: int = 2

    fsdp_config: Optional[Dict[str, Any]] = None
    fsdp_config_path: Optional[str] = None

    gpu_monitor: bool = True
    gpu_log_dir: str = "gpu_logs"
    gpu_log_interval: int = 50

    model_name: str = ""
    data_path: str = ""
    output_dir: str = ""

    def __post_init__(self):
        self._validate()
        self._resolve_mode()
        self._compute_derived()

    def _validate(self):
        mode_values = [m.value for m in DistributedMode]
        if self.mode not in mode_values:
            raise ValueError(
                f"mode必须是{mode_values}之一, 当前: {self.mode}"
            )

        if self.models_per_gpu < 1:
            raise ValueError(
                f"models_per_gpu必须>=1, 当前: {self.models_per_gpu}"
            )

        if self.gpu_ids is not None:
            available = torch.cuda.device_count()
            for gid in self.gpu_ids:
                if gid < 0 or gid >= available:
                    raise ValueError(
                        f"gpu_ids中{gid}超出范围(0-{available-1}), 可用GPU数: {available}"
                    )

        if self.gpu_groups is not None:
            all_gpus = []
            for group in self.gpu_groups:
                if len(group) < 1:
                    raise ValueError("GPU分组不能为空")
                all_gpus.extend(group)
            if len(all_gpus) != len(set(all_gpus)):
                raise ValueError(
                    "GPU分组中存在重复GPU, 每个GPU只能属于一个分组"
                )
            available = torch.cuda.device_count()
            for gid in all_gpus:
                if gid < 0 or gid >= available:
                    raise ValueError(
                        f"gpu_groups中{gid}超出范围(0-{available-1})"
                    )

        if self.mode == "ddp" and self.device_map_strategy is not None:
            raise ValueError(
                "DDP模式与device_map互斥: DDP每进程加载完整模型, device_map会触发模型并行导致冲突"
                "如需模型并行请使用mode='device_map'"
            )

        if self.mode == "device_map" and self.device_map_strategy is None and self.custom_device_map is None:
            if self.gpu_groups is None:
                raise ValueError(
                    "device_map模式需要指定device_map_strategy或custom_device_map"
                )

        if self.mode == "fsdp" and self.device_map_strategy is not None:
            raise ValueError(
                "FSDP模式与device_map互斥: FSDP通过分片实现模型并行, 不需要device_map"
            )

        if self.gpu_ids is not None and self.gpu_groups is not None:
            overlap = set(self.gpu_ids) & set(g for g in self.gpu_groups for g in g)
            if overlap:
                raise ValueError(
                    f"gpu_ids和gpu_groups存在重叠GPU: {overlap}"
                )

        if self.models_per_gpu > 1 and self.mode == "device_map":
            logger.warning(
                "device_map模式下models_per_gpu>1无实际意义(模型已分片到多卡), 已自动设为1"
            )
            self.models_per_gpu = 1

        lr_values = [s.value for s in LRScalingStrategy]
        if self.lr_scaling not in lr_values:
            raise ValueError(
                f"lr_scaling必须是{lr_values}之一, 当前: {self.lr_scaling}"
            )

        valid_image_load_modes = {"preload", "lazy", "batch"}
        if self.image_load_mode not in valid_image_load_modes:
            raise ValueError(
                f"image_load_mode必须是{valid_image_load_modes}之一, 当前: {self.image_load_mode}"
            )

        if self.dataloader_num_workers is not None and self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers必须>=0")

        if self.dataloader_prefetch_factor is not None and self.dataloader_prefetch_factor < 1:
            raise ValueError("dataloader_prefetch_factor必须>=1")

        if self.cpu_threads_per_rank is not None and self.cpu_threads_per_rank < 1:
            raise ValueError("cpu_threads_per_rank必须>=1")

        if self.attn_implementation is not None:
            valid_attn = {"sdpa", "flash_attention_2", "eager"}
            if self.attn_implementation not in valid_attn:
                raise ValueError(
                    f"attn_implementation必须是{valid_attn}之一, 当前: {self.attn_implementation}"
                )

    def _resolve_mode(self):
        is_distributed_env = os.environ.get("LOCAL_RANK") is not None

        if self.mode == "single_gpu" and is_distributed_env:
            logger.warning(
                "检测到分布式环境(LOCAL_RANK已设置), 但配置为single_gpu模式, 请确认是否正确"
            )

        if self.mode == "ddp" and not is_distributed_env and self.gpu_ids is not None and len(self.gpu_ids) > 1:
            logger.info(
                "DDP模式需要torchrun多进程启动, 当前为单进程环境(Notebook)"
                "实际将退化为单GPU模式, 请使用torchrun启动train_distributed.py"
            )

        if self.gpu_groups is not None and self.mode not in ("device_map", "single_gpu"):
            if self.mode == "ddp":
                logger.warning(
                    "DDP模式不支持gpu_groups(DDP每卡独立完整模型), 已切换为device_map模式"
                )
                self.mode = "device_map"

        if self.mode == "device_map" and self.gpu_groups is not None:
            self._num_data_parallel_groups = len(self.gpu_groups)
            self._gpus_per_model = len(self.gpu_groups[0])
        elif self.mode == "ddp":
            if self.gpu_ids is not None:
                self._num_data_parallel_groups = len(self.gpu_ids)
            else:
                self._num_data_parallel_groups = torch.cuda.device_count()
            self._gpus_per_model = 1
        elif self.mode == "fsdp":
            if self.gpu_ids is not None:
                self._num_data_parallel_groups = len(self.gpu_ids)
            else:
                self._num_data_parallel_groups = torch.cuda.device_count()
            self._gpus_per_model = 1
        else:
            self._num_data_parallel_groups = 1
            self._gpus_per_model = 1

    def _compute_derived(self):
        self._effective_models_per_gpu = self.models_per_gpu if self.mode == "ddp" else 1

        base_effective = self.per_device_batch_size * self.gradient_accumulation_steps
        self._effective_per_device_batch = base_effective * self._effective_models_per_gpu

        dp_groups = self._num_data_parallel_groups
        self._effective_global_batch = self._effective_per_device_batch * dp_groups

        self._effective_lr = self._scale_lr(self.learning_rate, dp_groups, self.lr_scaling)

        self._total_parallel_backward = dp_groups * self._effective_models_per_gpu

    @staticmethod
    def _scale_lr(base_lr: float, world_size: int, strategy: str) -> float:
        if strategy == "none" or world_size <= 1:
            return base_lr
        elif strategy == "linear":
            return base_lr * world_size
        elif strategy == "sqrt":
            return base_lr * (world_size**0.5)
        return base_lr

    @property
    def effective_global_batch(self) -> int:
        return self._effective_global_batch

    @property
    def effective_lr(self) -> float:
        return self._effective_lr

    @property
    def total_parallel_backward(self) -> int:
        return self._total_parallel_backward

    @property
    def num_data_parallel_groups(self) -> int:
        return self._num_data_parallel_groups

    @property
    def gpus_per_model(self) -> int:
        return self._gpus_per_model

    def get_device_map(self, local_rank: int = 0) -> Optional[Union[Dict, str]]:
        """根据配置生成device_map参数

        Args:
            local_rank: 当前进程的local_rank (用于gpu_groups模式确定组内GPU)

        Returns:
            device_map参数, 传给FastVisionModel.from_pretrained()
        """
        if self.mode == "ddp":
            is_distributed = os.environ.get("LOCAL_RANK") is not None
            if is_distributed:
                return None
            target_gpu = 0
            if self.gpu_ids is not None:
                target_gpu = self.gpu_ids[0]
            single_gpu_map = {"": target_gpu}
            return single_gpu_map

        if self.mode == "fsdp":
            return None

        if self.mode == "single_gpu":
            target_gpu = 0
            if self.gpu_ids is not None and len(self.gpu_ids) > 0:
                target_gpu = self.gpu_ids[0]
            single_gpu_map = {"": target_gpu}
            return single_gpu_map

        if self.mode == "device_map":
            if self.custom_device_map is not None:
                return self.custom_device_map

            if self.gpu_groups is not None:
                group = self.gpu_groups[local_rank]
                group_device_map = {}
                if self.device_map_strategy and self.device_map_strategy != DeviceMapStrategy.CUSTOM.value:
                    group_device_map = self.device_map_strategy
                else:
                    group_device_map = DeviceMapStrategy.BALANCED.value

                max_mem = self.max_memory_per_gpu
                if max_mem is None:
                    max_mem = {}
                    for gid in group:
                        props = torch.cuda.get_device_properties(gid)
                        max_mem[gid] = f"{int(props.total_memory / 1024**3 * 0.85)}GiB"

                return {
                    "strategy": group_device_map,
                    "gpu_group": group,
                    "max_memory": max_mem,
                }

            if self.device_map_strategy is not None:
                return self.device_map_strategy

            balanced = DeviceMapStrategy.BALANCED.value
            return balanced

        return None

    def get_cuda_visible_devices(self) -> str:
        """生成CUDA_VISIBLE_DEVICES环境变量值

        Returns:
            CUDA_VISIBLE_DEVICES字符串, 如"0,1,2,3,4,5,6,7"
        """
        if self.gpu_ids is not None:
            return ",".join(str(g) for g in self.gpu_ids)

        if self.gpu_groups is not None:
            all_gpus = []
            for group in self.gpu_groups:
                all_gpus.extend(group)
            return ",".join(str(g) for g in sorted(all_gpus))

        n = torch.cuda.device_count()
        return ",".join(str(i) for i in range(n))

    def get_torchrun_command(self, script_path: str = "train_distributed.py") -> str:
        """生成torchrun启动命令

        Args:
            script_path: 训练脚本路径

        Returns:
            完整的torchrun命令字符串
        """
        cuda_devices = self.get_cuda_visible_devices()

        if self.mode == "ddp":
            nproc = self.num_data_parallel_groups
            cmd = f"CUDA_VISIBLE_DEVICES={cuda_devices} torchrun --nproc_per_node={nproc}"
            cmd += f" {script_path}"
            cmd += self._build_args_str()
            return cmd

        if self.mode == "device_map" and self.gpu_groups is not None:
            nproc = self.num_data_parallel_groups
            cmd = f"CUDA_VISIBLE_DEVICES={cuda_devices} torchrun --nproc_per_node={nproc}"
            cmd += f" {script_path}"
            cmd += self._build_args_str()
            cmd += " --use_ddp"
            cmd += f" --device_map {self.device_map_strategy or 'balanced'}"
            cmd += f" --gpu_groups '{json.dumps(self.gpu_groups)}'"
            return cmd

        if self.mode == "fsdp":
            nproc = self.num_data_parallel_groups
            cmd = f"CUDA_VISIBLE_DEVICES={cuda_devices} torchrun --nproc_per_node={nproc}"
            cmd += f" {script_path}"
            cmd += self._build_args_str()
            cmd += " --use_fsdp"
            return cmd

        if self.mode == "single_gpu":
            gpu_list = cuda_devices
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_list} python {script_path}"
            cmd += self._build_args_str()
            return cmd

        return f"python {script_path}"

    def _build_args_str(self) -> str:
        args = []
        if self.model_name:
            args.append(f"--model_name {self.model_name}")
        if self.data_path:
            args.append(f"--data_path {self.data_path}")
        if self.output_dir:
            args.append(f"--output_dir {self.output_dir}")
        args.append(f"--per_device_batch_size {self.per_device_batch_size}")
        args.append(f"--gradient_accumulation_steps {self.gradient_accumulation_steps}")
        args.append(f"--learning_rate {self.learning_rate}")
        args.append(f"--lr_scaling {self.lr_scaling}")
        args.append(f"--max_seq_length {self.max_seq_length}")
        args.append(f"--num_epochs {self.num_epochs}")
        args.append(f"--warmup_ratio {self.warmup_ratio}")
        args.append(f"--lora_r {self.lora_r}")
        args.append(f"--lora_alpha {self.lora_alpha}")
        if self.bf16:
            args.append("--bf16")
        if self.vision_mode:
            args.append("--vision_mode")
        if self.load_in_4bit:
            args.append("--load_in_4bit")
        if self.tf32:
            args.append("--tf32")
        if self.dataloader_num_workers is not None:
            args.append(f"--dataloader_num_workers {self.dataloader_num_workers}")
        if self.dataloader_prefetch_factor is not None:
            args.append(f"--dataloader_prefetch_factor {self.dataloader_prefetch_factor}")
        if self.dataloader_pin_memory:
            args.append("--dataloader_pin_memory")
        if self.dataloader_persistent_workers:
            args.append("--dataloader_persistent_workers")
        if self.cpu_threads_per_rank is not None:
            args.append(f"--cpu_threads_per_rank {self.cpu_threads_per_rank}")
        args.append(f"--image_load_mode {self.image_load_mode}")
        if self.image_batch_size is not None:
            args.append(f"--image_batch_size {self.image_batch_size}")
        if self.materialize_vision_dataset:
            args.append("--materialize_vision_dataset")
        if self.attn_implementation is not None:
            args.append(f"--attn_implementation {self.attn_implementation}")
        if self.gpu_monitor:
            args.append("--gpu_monitor")
            args.append(f"--gpu_log_dir {self.gpu_log_dir}")
            args.append(f"--gpu_log_interval {self.gpu_log_interval}")
        return " " + " ".join(args)

    def get_training_kwargs(self) -> Dict[str, Any]:
        """生成HuggingFace SFTConfig/training_args参数字典

        Returns:
            可直接传给SFTConfig的参数字典
        """
        use_bf16 = self.bf16 and torch.cuda.is_bf16_supported()
        use_fp16 = self.fp16 or (not use_bf16 and not self.bf16)

        dataset_len = 1000
        warmup_steps = max(1, int(dataset_len * self.num_epochs / self.effective_global_batch * self.warmup_ratio))

        effective_batch_size = self.per_device_batch_size
        if self.mode == "ddp" and self.models_per_gpu > 1:
            effective_batch_size = self.per_device_batch_size * self.models_per_gpu

        kwargs = {
            "output_dir": self.output_dir or "outputs",
            "per_device_train_batch_size": effective_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.effective_lr,
            "num_train_epochs": self.num_epochs,
            "warmup_steps": warmup_steps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "optim": self.optim,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "seed": self.seed,
            "bf16": use_bf16,
            "fp16": use_fp16,
            "max_seq_length": self.max_seq_length,
            "packing": False,
            "report_to": "none",
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "dataloader_drop_last": self.dataloader_drop_last,
        }

        if self.vision_mode:
            kwargs["remove_unused_columns"] = False
            kwargs["dataset_text_field"] = ""

        if self.dataloader_num_workers is not None:
            kwargs["dataloader_num_workers"] = self.dataloader_num_workers
            if self.dataloader_num_workers > 0:
                kwargs["dataloader_persistent_workers"] = self.dataloader_persistent_workers
                if self.dataloader_prefetch_factor is not None:
                    kwargs["dataloader_prefetch_factor"] = self.dataloader_prefetch_factor

        if self.mode == "ddp":
            kwargs["ddp_find_unused_parameters"] = self.ddp_find_unused_parameters

        if self.mode == "fsdp":
            fsdp_cfg = self._load_fsdp_config()
            kwargs["fsdp_config"] = fsdp_cfg

        return kwargs

    def _load_fsdp_config(self) -> Dict[str, Any]:
        if self.fsdp_config is not None:
            return self.fsdp_config

        if self.fsdp_config_path is not None:
            path = Path(self.fsdp_config_path)
            if path.exists():
                with open(path, "r") as f:
                    return json.load(f)

        default_path = Path(__file__).resolve().parents[4] / "configs" / "training" / "fsdp_config.json"
        if default_path.exists():
            with open(default_path, "r") as f:
                return json.load(f)

        return {
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_sharding_strategy": "FULL_SHARD",
            "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
            "fsdp_use_orig_params": True,
            "fsdp_sync_module_states": True,
        }

    def get_model_kwargs(self) -> Dict[str, Any]:
        """生成FastVisionModel.from_pretrained()参数字典

        Returns:
            可直接传给from_pretrained的参数字典
        """
        kwargs = {
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,
            "dtype": None,
        }

        if self.load_in_4bit:
            kwargs["load_in_4bit"] = True

        device_map = self.get_device_map()
        if device_map is not None:
            if isinstance(device_map, dict) and "strategy" in device_map:
                strategy = device_map["strategy"]
                gpu_group = device_map.get("gpu_group")
                max_memory = device_map.get("max_memory")

                kwargs["device_map"] = strategy

                mapping = getattr(self, "_gpu_group_mapping", None)
                if mapping is not None and gpu_group is not None and max_memory is not None:
                    original_group, remapped_group = mapping
                    remapped_max_memory = {}
                    for i, remapped_id in enumerate(remapped_group):
                        original_id = original_group[i]
                        if original_id in max_memory:
                            remapped_max_memory[remapped_id] = max_memory[original_id]
                    kwargs["max_memory"] = remapped_max_memory
                elif max_memory is not None:
                    kwargs["max_memory"] = max_memory
            else:
                kwargs["device_map"] = device_map

        if self.vision_mode:
            kwargs["disable_log_stats"] = True

        if self.attn_implementation is not None:
            kwargs["attn_implementation"] = self.attn_implementation

        return kwargs

    def summary(self) -> str:
        """生成配置摘要字符串

        Returns:
            格式化的配置摘要, 适合打印或日志
        """
        lines = []
        sep = "=" * 70
        lines.append(sep)
        lines.append("分布式训练配置摘要")
        lines.append(sep)

        lines.append(f"模式: {self.mode}")
        lines.append(f"数据并行组数: {self.num_data_parallel_groups}")
        lines.append(f"每组GPU数: {self.gpus_per_model}")
        if self.mode == "ddp":
            lines.append(f"每GPU模型倍数(models_per_gpu): {self.models_per_gpu}")
        lines.append(f"总并行反向传播路数: {self.total_parallel_backward}")

        lines.append("")
        lines.append("训练参数:")
        lines.append(f"  每GPU批次: {self.per_device_batch_size}")
        if self.mode == "ddp" and self.models_per_gpu > 1:
            lines.append(f"  实际每GPU批次(×models_per_gpu): {self.per_device_batch_size * self.models_per_gpu}")
        lines.append(f"  梯度累积: {self.gradient_accumulation_steps}")
        lines.append(f"  有效全局批次: {self.effective_global_batch}")
        lines.append(f"  基础学习率: {self.learning_rate}")
        lines.append(f"  有效学习率({self.lr_scaling}缩放): {self.effective_lr:.6f}")
        lines.append(f"  混合精度: {'BF16' if self.bf16 else 'FP16' if self.fp16 else 'FP32'}")
        lines.append(f"  优化器: {self.optim}")
        lines.append(f"  LoRA: r={self.lora_r}, alpha={self.lora_alpha}")
        lines.append(f"  TF32: {self.tf32}")
        lines.append(f"  DataLoader workers: {self.dataloader_num_workers if self.dataloader_num_workers is not None else 'auto'}")
        lines.append(f"  Prefetch factor: {self.dataloader_prefetch_factor if self.dataloader_prefetch_factor is not None else 'auto'}")
        lines.append(f"  CPU线程/Rank: {self.cpu_threads_per_rank if self.cpu_threads_per_rank is not None else 'auto'}")
        lines.append(f"  图片加载模式: {self.image_load_mode}")
        lines.append(f"  数据集预物化: {self.materialize_vision_dataset}")
        lines.append(f"  注意力实现: {self.attn_implementation or 'auto (Unsloth默认)'}")

        lines.append("")
        lines.append("硬件配置:")
        if self.gpu_ids is not None:
            lines.append(f"  指定GPU: {self.gpu_ids}")
        elif self.gpu_groups is not None:
            lines.append(f"  GPU分组: {self.gpu_groups}")
        else:
            n = torch.cuda.device_count()
            lines.append(f"  可用GPU: {n}")

        device_map = self.get_device_map()
        dm_desc = str(device_map) if device_map is not None else "None (DDP每进程独立GPU)"
        lines.append(f"  device_map: {dm_desc}")

        lines.append("")
        lines.append("启动命令:")
        lines.append(f"  {self.get_torchrun_command()}")

        lines.append(sep)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典, 用于JSON保存或参数传递"""
        result = {
            "mode": self.mode,
            "gpu_ids": self.gpu_ids,
            "models_per_gpu": self.models_per_gpu,
            "gpu_groups": self.gpu_groups,
            "device_map_strategy": self.device_map_strategy,
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "lr_scaling": self.lr_scaling,
            "effective_global_batch": self.effective_global_batch,
            "effective_lr": self.effective_lr,
            "total_parallel_backward": self.total_parallel_backward,
            "num_data_parallel_groups": self.num_data_parallel_groups,
            "gpus_per_model": self.gpus_per_model,
            "bf16": self.bf16,
            "vision_mode": self.vision_mode,
            "tf32": self.tf32,
            "load_in_4bit": self.load_in_4bit,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "optim": self.optim,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_prefetch_factor": self.dataloader_prefetch_factor,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "dataloader_persistent_workers": self.dataloader_persistent_workers,
            "ddp_find_unused_parameters": self.ddp_find_unused_parameters,
            "cpu_threads_per_rank": self.cpu_threads_per_rank,
            "image_load_mode": self.image_load_mode,
            "image_batch_size": self.image_batch_size,
            "materialize_vision_dataset": self.materialize_vision_dataset,
            "attn_implementation": self.attn_implementation,
            "max_seq_length": self.max_seq_length,
            "num_epochs": self.num_epochs,
            "model_name": self.model_name,
            "data_path": self.data_path,
            "output_dir": self.output_dir,
        }
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DistributedConfig":
        """从字典反序列化创建配置"""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> "DistributedConfig":
        """从JSON文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: str) -> None:
        """保存配置到JSON文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def create_ddp_config(
    gpu_ids: Optional[List[int]] = None,
    models_per_gpu: int = 1,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 4e-5,
    lr_scaling: str = "linear",
    **kwargs,
) -> DistributedConfig:
    """便捷函数: 创建DDP数据并行配置

    适用于小模型(单卡可容纳完整模型), 通过models_per_gpu倍增吞吐量.

    Args:
        gpu_ids: 参与训练的GPU列表, None则使用所有可用GPU
        models_per_gpu: 每GPU吞吐量倍数, 映射到batch_size缩放
            例: models_per_gpu=2, per_device_batch_size=4 → 实际每GPU批次=8
            例: 8卡×2倍 = 16路并行反向传播
        per_device_batch_size: 基础每GPU批次大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 基础学习率
        lr_scaling: 学习率缩放策略 (none/linear/sqrt)

    Returns:
        DistributedConfig实例
    """
    return DistributedConfig(
        mode="ddp",
        gpu_ids=gpu_ids,
        models_per_gpu=models_per_gpu,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scaling=lr_scaling,
        **kwargs,
    )


def create_device_map_config(
    gpu_groups: Optional[List[List[int]]] = None,
    device_map_strategy: str = "balanced",
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 4e-5,
    lr_scaling: str = "linear",
    max_memory_per_gpu: Optional[Dict[int, str]] = None,
    custom_device_map: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> DistributedConfig:
    """便捷函数: 创建device_map模型并行配置

    适用于大模型(需多卡共同容纳), 支持GPU分组实现2D并行.

    Args:
        gpu_groups: GPU分组配置, 每组承载1个完整模型
            例: [[0,1], [2,3]] → 2组×2卡, 2路数据并行×2卡模型并行
            例: [[0,1,2,3], [4,5,6,7]] → 2组×4卡, 2路数据并行×4卡模型并行
        device_map_strategy: 模型分片策略 (balanced/auto/balanced_low_0)
        per_device_batch_size: 每数据并行组批次大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 基础学习率
        lr_scaling: 学习率缩放策略
        max_memory_per_gpu: 每GPU最大可用显存, 如{0: "40GiB", 1: "40GiB"}
        custom_device_map: 自定义device_map字典 (覆盖strategy)

    Returns:
        DistributedConfig实例
    """
    return DistributedConfig(
        mode="device_map",
        gpu_groups=gpu_groups,
        device_map_strategy=device_map_strategy,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scaling=lr_scaling,
        max_memory_per_gpu=max_memory_per_gpu,
        custom_device_map=custom_device_map,
        **kwargs,
    )


def create_fsdp_config(
    gpu_ids: Optional[List[int]] = None,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 4e-5,
    lr_scaling: str = "linear",
    fsdp_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> DistributedConfig:
    """便捷函数: 创建FSDP分片并行配置

    适用于大模型(31B+), 参数/梯度/优化器全分片.

    Args:
        gpu_ids: 参与训练的GPU列表
        per_device_batch_size: 每GPU批次大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 基础学习率
        lr_scaling: 学习率缩放策略
        fsdp_config: FSDP配置字典

    Returns:
        DistributedConfig实例
    """
    return DistributedConfig(
        mode="fsdp",
        gpu_ids=gpu_ids,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scaling=lr_scaling,
        fsdp_config=fsdp_config,
        **kwargs,
    )


def auto_detect_config(
    model_vram_gb: float = None,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 4e-5,
    **kwargs,
) -> DistributedConfig:
    """自动检测并创建最优分布式配置

    根据模型显存需求和可用GPU资源, 自动选择DDP/device_map/FSDP模式:
      - 模型可放入单卡 → DDP (最低通信开销)
      - 模型需N卡容纳 → device_map with gpu_groups (N卡/组, 余下GPU做数据并行)
      - 模型极大, 需全部GPU分片 → FSDP

    Args:
        model_vram_gb: 模型所需显存(GB), None则默认小模型(10GB)
        per_device_batch_size: 每GPU批次大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 基础学习率

    Returns:
        最优DistributedConfig实例
    """
    if model_vram_gb is None:
        model_vram_gb = 10.0

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("未检测到GPU, 无法配置分布式训练")

    gpu_vrams = [
        torch.cuda.get_device_properties(i).total_memory / 1024**3
        for i in range(n_gpus)
    ]
    min_gpu_vram = min(gpu_vrams)
    usable_vram = min_gpu_vram * 0.80

    if model_vram_gb <= usable_vram:
        models_fit = int(usable_vram / model_vram_gb)
        models_per_gpu = min(models_fit, 2)
        logger.info(
            f"模型{model_vram_gb:.1f}GB可放入单卡(可用{usable_vram:.1f}GB), "
            f"选择DDP模式, models_per_gpu={models_per_gpu}"
        )
        return create_ddp_config(
            models_per_gpu=models_per_gpu,
            per_device_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            **kwargs,
        )

    gpus_needed = int(model_vram_gb / usable_vram) + 1
    if gpus_needed > n_gpus:
        logger.info(
            f"模型{model_vram_gb:.1f}GB需{gpus_needed}卡但仅{n_gpus}卡可用, 选择FSDP全分片"
        )
        return create_fsdp_config(
            per_device_batch_size=max(1, per_device_batch_size // 2),
            gradient_accumulation_steps=gradient_accumulation_steps * 2,
            learning_rate=learning_rate,
            **kwargs,
        )

    num_groups = n_gpus // gpus_needed
    remainder = n_gpus % gpus_needed
    if remainder != 0:
        logger.warning(
            f"{n_gpus}卡不能被{gpus_needed}均匀分组, {remainder}卡将闲置"
        )

    groups = []
    for i in range(num_groups):
        start = i * gpus_needed
        group = list(range(start, start + gpus_needed))
        groups.append(group)

    logger.info(
        f"模型{model_vram_gb:.1f}GB需{gpus_needed}卡/组, "
        f"{n_gpus}卡分{num_groups}组, device_map+DDP 2D并行"
    )
    return create_device_map_config(
        gpu_groups=groups,
        device_map_strategy="balanced",
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        **kwargs,
    )
