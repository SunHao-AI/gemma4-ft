"""
模型配置加载和验证工具

提供统一的配置管理接口，支持:
- YAML配置文件加载
- 配置验证和完整性检查
- 配置参数访问
- 配置导出为字典
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ConfigValidationError(Exception):
    """配置验证错误"""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"配置验证失败:\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass
class ModelConfig:
    """模型配置数据类"""

    name: str = ""
    family: str = ""
    version: str = ""
    base_model_path: str = ""
    description: str = ""


@dataclass
class DataPreparationConfig:
    """数据准备配置"""

    coord_norm: str = "norm_1000"
    coord_format: str = "xyxy"
    output_format: str = "box_2d_json"
    prompt_lang: str = "zh"
    prompt_style: str = "descriptive"
    split_ratio: str = "8:1:1"
    split_method: str = "random"
    split_seed: Optional[int] = None
    class_whitelist: List[str] = field(default_factory=list)
    class_blacklist: List[str] = field(default_factory=list)
    class_remap: Dict[str, str] = field(default_factory=dict)
    shape_types: List[str] = field(default_factory=lambda: ["rectangle", "polygon"])
    min_bbox_size: int = 2
    keep_empty: bool = False


@dataclass
class ModelLoadingConfig:
    """模型加载配置"""

    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    attention_implementation: str = "sdpa"
    use_cache: bool = True
    trust_remote_code: bool = False


@dataclass
class LoRAConfig:
    """LoRA配置"""

    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    bias: str = "none"
    use_rslora: bool = False


@dataclass
class TrainingConfig:
    """训练配置"""

    num_epochs: int = 1
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    lr_scaling: str = "sqrt"
    warmup_ratio: float = 0.1
    optimizer_type: str = "adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    vision_enabled: bool = True
    image_width: int = 896
    image_height: int = 896
    image_load_mode: str = "lazy"


@dataclass
class DistributedConfig:
    """分布式训练配置"""

    mode: str = "ddp"
    num_gpus: int = 8
    ddp_backend: str = "nccl"
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_auto_wrap_policy: str = "TRANSFORMER_BASED_WRAP"
    fsdp_backward_prefetch: str = "BACKWARD_PRE"
    fsdp_cpu_offload: bool = False
    fsdp_min_num_params: int = 1000


@dataclass
class DataLoaderConfig:
    """数据加载配置"""

    num_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = False


@dataclass
class OutputConfig:
    """输出配置"""

    base_dir: str = "models/finetuned"
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    log_level: str = "INFO"
    log_steps: int = 10
    report_to: str = "tensorboard"
    lora_adapter_subdir: str = ""
    lora_adapter_use_latest: bool = True
    lora_adapter_timestamp: str = ""


@dataclass
class EvaluationConfig:
    """评估配置"""

    eval_strategy: str = "steps"
    eval_steps: int = 100
    eval_on_start: bool = False
    metrics: List[str] = field(default_factory=lambda: ["loss"])
    post_train_eval_auto_run: bool = False
    post_train_eval_raise_on_error: bool = False


@dataclass
class InferenceConfig:
    """推理配置"""

    max_new_tokens: int = 128
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True


VALID_COORD_NORMS: Set[str] = {"raw", "norm_1", "norm_100", "norm_1000"}
VALID_COORD_FORMATS: Set[str] = {"xyxy", "yxyx", "xywh", "cxcywh"}
VALID_OUTPUT_FORMATS: Set[str] = {"labelme_text", "box_2d_json"}
VALID_PROMPT_LANGS: Set[str] = {"en", "zh"}
VALID_PROMPT_STYLES: Set[str] = {"simple", "descriptive", "cot"}
VALID_SPLIT_METHODS: Set[str] = {"random", "sequential", "stratified"}
VALID_ATTENTION_IMPLS: Set[str] = {"sdpa", "flash_attention_2", "eager"}
VALID_DISTRIBUTED_MODES: Set[str] = {"single", "DDP", "FSDP", "device_map", "auto", "multi_node", "compare"}
VALID_LR_SCALINGS: Set[str] = {"none", "linear", "sqrt"}
VALID_IMAGE_LOAD_MODES: Set[str] = {"lazy", "preload"}

GEMMA4_REQUIRED_CONFIG: Dict[str, Any] = {
    "coord_norm": "norm_1000",
    "coord_format": "yxyx",
}


class TrainingConfigLoader:
    """训练配置加载器"""

    CONFIG_DIR: Path = Path(__file__).parent
    BASE_CONFIG_FILE: str = "base_config.yaml"

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self._raw_config: Dict[str, Any] = {}
        self._config_path: Optional[Path] = None
        self._validation_errors: List[str] = []

        self.model: ModelConfig = ModelConfig()
        self.data_preparation: DataPreparationConfig = DataPreparationConfig()
        self.model_loading: ModelLoadingConfig = ModelLoadingConfig()
        self.lora: LoRAConfig = LoRAConfig()
        self.training: TrainingConfig = TrainingConfig()
        self.distributed: DistributedConfig = DistributedConfig()
        self.dataloader: DataLoaderConfig = DataLoaderConfig()
        self.output: OutputConfig = OutputConfig()
        self.evaluation: EvaluationConfig = EvaluationConfig()
        self.inference: InferenceConfig = InferenceConfig()

        if config_path:
            self.load(config_path)

    @classmethod
    def check_yaml_available(cls) -> bool:
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML 未安装。请执行: pip install pyyaml\n" "或者使用 JSON 格式配置文件。")
        return True

    @classmethod
    def get_available_configs(cls) -> List[str]:
        available: List[str] = []
        for f in cls.CONFIG_DIR.glob("*_config.yaml"):
            available.append(f.stem)
        return sorted(available)

    @classmethod
    def get_config_path(cls, model_name: str) -> Path:
        config_file = f"{model_name}_config.yaml"
        path = cls.CONFIG_DIR / config_file
        if not path.exists():
            available = cls.get_available_configs()
            raise FileNotFoundError(f"配置文件不存在: {config_file}\n" f"可用配置: {available}")
        return path

    def load(self, config_path: Union[str, Path]) -> "TrainingConfigLoader":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        self._config_path = path

        if path.suffix in (".yaml", ".yml"):
            self.check_yaml_available()
            with open(path, "r", encoding="utf-8") as f:
                self._raw_config = yaml.safe_load(f) or {}
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                self._raw_config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {path.suffix}")

        self._parse_config()
        return self

    def load_by_name(self, model_name: str) -> "TrainingConfigLoader":
        path = self.get_config_path(model_name)
        return self.load(path)

    def _parse_config(self) -> None:
        self._parse_model_config()
        self._parse_data_preparation_config()
        self._parse_model_loading_config()
        self._parse_lora_config()
        self._parse_training_config()
        self._parse_distributed_config()
        self._parse_dataloader_config()
        self._parse_output_config()
        self._parse_evaluation_config()
        self._parse_inference_config()

    def _parse_model_config(self) -> None:
        cfg = self._raw_config.get("model", {})
        self.model = ModelConfig(
            name=cfg.get("name", ""),
            family=cfg.get("family", ""),
            version=cfg.get("version", ""),
            base_model_path=cfg.get("base_model_path", ""),
            description=cfg.get("description", ""),
        )

    def _parse_data_preparation_config(self) -> None:
        cfg = self._raw_config.get("data_preparation", {})
        prompt_cfg = cfg.get("prompt", {})
        split_cfg = cfg.get("split", {})
        filter_cfg = cfg.get("filter", {})

        self.data_preparation = DataPreparationConfig(
            coord_norm=cfg.get("coord_norm", "norm_1000"),
            coord_format=cfg.get("coord_format", "xyxy"),
            output_format=cfg.get("output_format", "box_2d_json"),
            prompt_lang=prompt_cfg.get("lang", "zh"),
            prompt_style=prompt_cfg.get("style", "descriptive"),
            split_ratio=split_cfg.get("ratio", "8:1:1"),
            split_method=split_cfg.get("method", "random"),
            split_seed=int(split_cfg.get("seed")) if split_cfg.get("seed") is not None else None,
            class_whitelist=filter_cfg.get("class_whitelist", []),
            class_blacklist=filter_cfg.get("class_blacklist", []),
            class_remap=filter_cfg.get("class_remap", {}),
            shape_types=filter_cfg.get("shape_types", ["rectangle", "polygon"]),
            min_bbox_size=int(filter_cfg.get("min_bbox_size", 2)),
            keep_empty=bool(filter_cfg.get("keep_empty", False)),
        )

    def _parse_model_loading_config(self) -> None:
        cfg = self._raw_config.get("model_loading", {})
        quant_cfg = cfg.get("quantization", {})
        attn_cfg = cfg.get("attention", {})
        opt_cfg = cfg.get("options", {})

        self.model_loading = ModelLoadingConfig(
            max_seq_length=int(cfg.get("max_seq_length", 2048)),
            load_in_4bit=bool(quant_cfg.get("load_in_4bit", True)),
            load_in_8bit=bool(quant_cfg.get("load_in_8bit", False)),
            attention_implementation=attn_cfg.get("implementation", "sdpa"),
            use_cache=bool(opt_cfg.get("use_cache", True)),
            trust_remote_code=bool(opt_cfg.get("trust_remote_code", False)),
        )

    def _parse_lora_config(self) -> None:
        cfg = self._raw_config.get("lora", {})

        self.lora = LoRAConfig(
            r=int(cfg.get("r", 16)),
            alpha=int(cfg.get("alpha", 16)),
            dropout=float(cfg.get("dropout", 0.0)),
            target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias=cfg.get("bias", "none"),
            use_rslora=bool(cfg.get("use_rslora", False)),
        )

    def _parse_training_config(self) -> None:
        cfg = self._raw_config.get("training", {})
        opt_cfg = cfg.get("optimizer", {})
        prec_cfg = cfg.get("precision", {})
        vision_cfg = cfg.get("vision", {})

        self.training = TrainingConfig(
            num_epochs=int(cfg.get("num_epochs", 1)),
            per_device_batch_size=int(cfg.get("per_device_batch_size", 8)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 2)),
            learning_rate=float(cfg.get("learning_rate", 2e-5)),
            lr_scaling=cfg.get("lr_scaling", "sqrt"),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
            optimizer_type=opt_cfg.get("type", "adamw_8bit"),
            weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
            max_grad_norm=float(opt_cfg.get("max_grad_norm", 1.0)),
            bf16=bool(prec_cfg.get("bf16", True)),
            fp16=bool(prec_cfg.get("fp16", False)),
            tf32=bool(prec_cfg.get("tf32", True)),
            vision_enabled=bool(vision_cfg.get("enabled", True)),
            image_width=int(vision_cfg.get("image_width", 896)),
            image_height=int(vision_cfg.get("image_height", 896)),
            image_load_mode=vision_cfg.get("image_load_mode", "lazy"),
        )

    def _parse_distributed_config(self) -> None:
        cfg = self._raw_config.get("distributed", {})
        ddp_cfg = cfg.get("ddp", {})
        fsdp_cfg = cfg.get("fsdp", {})

        self.distributed = DistributedConfig(
            mode=cfg.get("mode", "ddp"),
            num_gpus=int(cfg.get("num_gpus", 8)),
            ddp_backend=ddp_cfg.get("backend", "nccl"),
            fsdp_sharding_strategy=fsdp_cfg.get("sharding_strategy", "FULL_SHARD"),
            fsdp_auto_wrap_policy=fsdp_cfg.get("auto_wrap_policy", "TRANSFORMER_BASED_WRAP"),
            fsdp_backward_prefetch=fsdp_cfg.get("backward_prefetch", "BACKWARD_PRE"),
            fsdp_cpu_offload=bool(fsdp_cfg.get("cpu_offload", False)),
            fsdp_min_num_params=int(fsdp_cfg.get("min_num_params", 1000)),
        )

    def _parse_dataloader_config(self) -> None:
        cfg = self._raw_config.get("dataloader", {})

        self.dataloader = DataLoaderConfig(
            num_workers=int(cfg.get("num_workers", 4)),
            prefetch_factor=int(cfg.get("prefetch_factor", 4)),
            pin_memory=bool(cfg.get("pin_memory", True)),
            persistent_workers=bool(cfg.get("persistent_workers", True)),
            drop_last=bool(cfg.get("drop_last", False)),
        )

    def _parse_output_config(self) -> None:
        cfg = self._raw_config.get("output", {})
        log_cfg = cfg.get("logging", {})
        lora_cfg = cfg.get("lora_adapter", {})

        self.output = OutputConfig(
            base_dir=cfg.get("base_dir", "models/finetuned"),
            save_strategy=cfg.get("save_strategy", "steps"),
            save_steps=int(cfg.get("save_steps", 500)),
            save_total_limit=int(cfg.get("save_total_limit", 3)),
            log_level=log_cfg.get("log_level", "INFO"),
            log_steps=int(log_cfg.get("log_steps", 10)),
            report_to=log_cfg.get("report_to", "tensorboard"),
            lora_adapter_subdir=lora_cfg.get("subdir", f"{self.model.name}_lora" if self.model.name else ""),
            lora_adapter_use_latest=bool(lora_cfg.get("use_latest", True)),
            lora_adapter_timestamp=lora_cfg.get("timestamp", ""),
        )

    def _parse_evaluation_config(self) -> None:
        cfg = self._raw_config.get("evaluation", {})
        post_eval_cfg = cfg.get("post_training_eval", {})

        self.evaluation = EvaluationConfig(
            eval_strategy=cfg.get("eval_strategy", "steps"),
            eval_steps=int(cfg.get("eval_steps", 100)),
            eval_on_start=bool(cfg.get("eval_on_start", False)),
            metrics=cfg.get("metrics", ["loss"]),
            post_train_eval_auto_run=bool(post_eval_cfg.get("auto_run", False)),
            post_train_eval_raise_on_error=bool(post_eval_cfg.get("raise_on_error", False)),
        )

    def _parse_inference_config(self) -> None:
        cfg = self._raw_config.get("inference", {})

        self.inference = InferenceConfig(
            max_new_tokens=int(cfg.get("max_new_tokens", 128)),
            temperature=float(cfg.get("temperature", 0.1)),
            top_p=float(cfg.get("top_p", 0.9)),
            top_k=int(cfg.get("top_k", 50)),
            do_sample=bool(cfg.get("do_sample", True)),
        )

    def validate(self, strict: bool = True) -> Tuple[bool, List[str]]:
        self._validation_errors = []

        self._validate_data_preparation()
        self._validate_model_loading()
        self._validate_lora()
        self._validate_training()
        self._validate_distributed()
        self._validate_gemma4_specific()

        is_valid = len(self._validation_errors) == 0

        if strict and not is_valid:
            raise ConfigValidationError(self._validation_errors)

        return (is_valid, self._validation_errors)

    def _validate_data_preparation(self) -> None:
        dp = self.data_preparation

        if dp.coord_norm not in VALID_COORD_NORMS:
            self._validation_errors.append(f"无效的 coord_norm: '{dp.coord_norm}'，可选值: {VALID_COORD_NORMS}")

        if dp.coord_format not in VALID_COORD_FORMATS:
            self._validation_errors.append(f"无效的 coord_format: '{dp.coord_format}'，可选值: {VALID_COORD_FORMATS}")

        if dp.output_format not in VALID_OUTPUT_FORMATS:
            self._validation_errors.append(f"无效的 output_format: '{dp.output_format}'，可选值: {VALID_OUTPUT_FORMATS}")

        if dp.prompt_lang not in VALID_PROMPT_LANGS:
            self._validation_errors.append(f"无效的 prompt_lang: '{dp.prompt_lang}'，可选值: {VALID_PROMPT_LANGS}")

        if dp.prompt_style not in VALID_PROMPT_STYLES:
            self._validation_errors.append(f"无效的 prompt_style: '{dp.prompt_style}'，可选值: {VALID_PROMPT_STYLES}")

        if dp.split_method not in VALID_SPLIT_METHODS:
            self._validation_errors.append(f"无效的 split_method: '{dp.split_method}'，可选值: {VALID_SPLIT_METHODS}")

        parts = dp.split_ratio.split(":")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            self._validation_errors.append(f"无效的 split_ratio 格式: '{dp.split_ratio}'，应为 'N:M:K' 格式")

    def _validate_model_loading(self) -> None:
        ml = self.model_loading

        if ml.max_seq_length <= 0:
            self._validation_errors.append(f"max_seq_length 必须为正数: {ml.max_seq_length}")

        if ml.attention_implementation not in VALID_ATTENTION_IMPLS:
            self._validation_errors.append(f"无效的 attention_implementation: '{ml.attention_implementation}'，" f"可选值: {VALID_ATTENTION_IMPLS}")

        if ml.load_in_4bit and ml.load_in_8bit:
            self._validation_errors.append("不能同时启用 load_in_4bit 和 load_in_8bit")

    def _validate_lora(self) -> None:
        lora = self.lora

        if lora.r <= 0 or lora.r > 256:
            self._validation_errors.append(f"LoRA r 应在 1-256 范围内: {lora.r}")

        if lora.alpha <= 0:
            self._validation_errors.append(f"LoRA alpha 必须为正数: {lora.alpha}")

        if not 0 <= lora.dropout <= 1:
            self._validation_errors.append(f"LoRA dropout 应在 [0, 1] 范围内: {lora.dropout}")

        if not lora.target_modules:
            self._validation_errors.append("LoRA target_modules 不能为空")

    def _validate_training(self) -> None:
        tr = self.training

        if tr.num_epochs <= 0:
            self._validation_errors.append(f"num_epochs 必须为正数: {tr.num_epochs}")

        if tr.per_device_batch_size <= 0:
            self._validation_errors.append(f"per_device_batch_size 必须为正数: {tr.per_device_batch_size}")

        if tr.gradient_accumulation_steps <= 0:
            self._validation_errors.append(f"gradient_accumulation_steps 必须为正数: {tr.gradient_accumulation_steps}")

        if tr.learning_rate <= 0 or tr.learning_rate > 1e-2:
            self._validation_errors.append(f"learning_rate 应在 [1e-8, 1e-2] 范围内: {tr.learning_rate}")

        if tr.lr_scaling not in VALID_LR_SCALINGS:
            self._validation_errors.append(f"无效的 lr_scaling: '{tr.lr_scaling}'，可选值: {VALID_LR_SCALINGS}")

        if not 0 <= tr.warmup_ratio <= 1:
            self._validation_errors.append(f"warmup_ratio 应在 [0, 1] 范围内: {tr.warmup_ratio}")

        if tr.image_load_mode not in VALID_IMAGE_LOAD_MODES:
            self._validation_errors.append(f"无效的 image_load_mode: '{tr.image_load_mode}'，" f"可选值: {VALID_IMAGE_LOAD_MODES}")

    def _validate_distributed(self) -> None:
        dist = self.distributed

        if dist.mode.lower() not in VALID_DISTRIBUTED_MODES and dist.mode.upper() not in VALID_DISTRIBUTED_MODES:
            self._validation_errors.append(f"无效的 distributed mode: '{dist.mode}'，可选值: {VALID_DISTRIBUTED_MODES}")

        if dist.num_gpus <= 0:
            self._validation_errors.append(f"num_gpus 必须为正数: {dist.num_gpus}")

    def _validate_gemma4_specific(self) -> None:
        if self.model.family.lower() != "gemma":
            return

        dp = self.data_preparation

        if dp.coord_norm != GEMMA4_REQUIRED_CONFIG["coord_norm"]:
            self._validation_errors.append(f"Gemma4 模型要求 coord_norm='norm_1000'，" f"当前设置为 '{dp.coord_norm}'")

        if dp.coord_format != GEMMA4_REQUIRED_CONFIG["coord_format"]:
            self._validation_errors.append(f"Gemma4 模型要求 coord_format='yxyx'（box_2d为[y1,x1,y2,x2]），" f"当前设置为 '{dp.coord_format}'")

    def update(self, updates: Dict[str, Any]) -> "TrainingConfigLoader":
        for key, value in updates.items():
            parts = key.split(".", 1)
            if len(parts) == 1:
                self._raw_config[key] = value
            else:
                section, subkey = parts
                if section not in self._raw_config:
                    self._raw_config[section] = {}
                self._raw_config[section][subkey] = value

        self._parse_config()
        return self

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        result["model"] = {
            "name": self.model.name,
            "family": self.model.family,
            "version": self.model.version,
            "base_model_path": self.model.base_model_path,
            "description": self.model.description,
        }

        result["data_preparation"] = {
            "coord_norm": self.data_preparation.coord_norm,
            "coord_format": self.data_preparation.coord_format,
            "output_format": self.data_preparation.output_format,
            "prompt": {
                "lang": self.data_preparation.prompt_lang,
                "style": self.data_preparation.prompt_style,
            },
            "split": {
                "ratio": self.data_preparation.split_ratio,
                "method": self.data_preparation.split_method,
                "seed": self.data_preparation.split_seed,
            },
            "filter": {
                "class_whitelist": self.data_preparation.class_whitelist,
                "class_blacklist": self.data_preparation.class_blacklist,
                "class_remap": self.data_preparation.class_remap,
                "shape_types": self.data_preparation.shape_types,
                "min_bbox_size": self.data_preparation.min_bbox_size,
                "keep_empty": self.data_preparation.keep_empty,
            },
        }

        result["model_loading"] = {
            "max_seq_length": self.model_loading.max_seq_length,
            "quantization": {
                "load_in_4bit": self.model_loading.load_in_4bit,
                "load_in_8bit": self.model_loading.load_in_8bit,
            },
            "attention": {
                "implementation": self.model_loading.attention_implementation,
            },
            "options": {
                "use_cache": self.model_loading.use_cache,
                "trust_remote_code": self.model_loading.trust_remote_code,
            },
        }

        result["lora"] = {
            "r": self.lora.r,
            "alpha": self.lora.alpha,
            "dropout": self.lora.dropout,
            "target_modules": self.lora.target_modules,
            "bias": self.lora.bias,
            "use_rslora": self.lora.use_rslora,
        }

        result["training"] = {
            "num_epochs": self.training.num_epochs,
            "per_device_batch_size": self.training.per_device_batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "learning_rate": self.training.learning_rate,
            "lr_scaling": self.training.lr_scaling,
            "warmup_ratio": self.training.warmup_ratio,
            "optimizer": {
                "type": self.training.optimizer_type,
                "weight_decay": self.training.weight_decay,
                "max_grad_norm": self.training.max_grad_norm,
            },
            "precision": {
                "bf16": self.training.bf16,
                "fp16": self.training.fp16,
                "tf32": self.training.tf32,
            },
            "vision": {
                "enabled": self.training.vision_enabled,
                "image_width": self.training.image_width,
                "image_height": self.training.image_height,
                "image_load_mode": self.training.image_load_mode,
            },
        }

        result["distributed"] = {
            "mode": self.distributed.mode,
            "num_gpus": self.distributed.num_gpus,
            "ddp": {
                "backend": self.distributed.ddp_backend,
            },
            "fsdp": {
                "sharding_strategy": self.distributed.fsdp_sharding_strategy,
                "auto_wrap_policy": self.distributed.fsdp_auto_wrap_policy,
                "backward_prefetch": self.distributed.fsdp_backward_prefetch,
                "cpu_offload": self.distributed.fsdp_cpu_offload,
                "min_num_params": self.distributed.fsdp_min_num_params,
            },
        }

        result["dataloader"] = {
            "num_workers": self.dataloader.num_workers,
            "prefetch_factor": self.dataloader.prefetch_factor,
            "pin_memory": self.dataloader.pin_memory,
            "persistent_workers": self.dataloader.persistent_workers,
            "drop_last": self.dataloader.drop_last,
        }

        result["output"] = {
            "base_dir": self.output.base_dir,
            "save_strategy": self.output.save_strategy,
            "save_steps": self.output.save_steps,
            "save_total_limit": self.output.save_total_limit,
            "logging": {
                "log_level": self.output.log_level,
                "log_steps": self.output.log_steps,
                "report_to": self.output.report_to,
            },
            "lora_adapter": {
                "subdir": self.output.lora_adapter_subdir,
                "use_latest": self.output.lora_adapter_use_latest,
                "timestamp": self.output.lora_adapter_timestamp,
            },
        }

        result["evaluation"] = {
            "eval_strategy": self.evaluation.eval_strategy,
            "eval_steps": self.evaluation.eval_steps,
            "eval_on_start": self.evaluation.eval_on_start,
            "metrics": self.evaluation.metrics,
            "post_training_eval": {
                "auto_run": self.evaluation.post_train_eval_auto_run,
                "raise_on_error": self.evaluation.post_train_eval_raise_on_error,
            },
        }

        result["inference"] = {
            "max_new_tokens": self.inference.max_new_tokens,
            "temperature": self.inference.temperature,
            "top_p": self.inference.top_p,
            "top_k": self.inference.top_k,
            "do_sample": self.inference.do_sample,
        }

        return result

    def save(self, output_path: Union[str, Path], format: str = "yaml") -> None:
        path = Path(output_path)
        data = self.to_dict()

        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "yaml":
            self.check_yaml_available()
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        elif format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的格式: {format}")

    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        obj: Any = self

        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return default

        return obj

    def __repr__(self) -> str:
        model_info = f"{self.model.name}" if self.model.name else "未指定模型"
        coord_info = f"coord={self.data_preparation.coord_format}/{self.data_preparation.coord_norm}"
        return f"TrainingConfigLoader({model_info}, {coord_info})"

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"模型配置摘要: {self.model.name or '未指定'}",
            "=" * 60,
            "",
            "【模型信息】",
            f"  名称: {self.model.name}",
            f"  家族: {self.model.family}",
            f"  版本: {self.model.version}",
            "",
            "【数据准备】",
            f"  坐标归一化: {self.data_preparation.coord_norm}",
            f"  坐标格式: {self.data_preparation.coord_format}",
            f"  输出格式: {self.data_preparation.output_format}",
            f"  Prompt语言: {self.data_preparation.prompt_lang}",
            f"  数据划分: {self.data_preparation.split_ratio} ({self.data_preparation.split_method})",
            "",
            "【模型加载】",
            f"  最大序列长度: {self.model_loading.max_seq_length}",
            f"  量化模式: {'4-bit' if self.model_loading.load_in_4bit else '8-bit' if self.model_loading.load_in_8bit else 'FP16'}",
            f"  注意力实现: {self.model_loading.attention_implementation}",
            "",
            "【LoRA配置】",
            f"  Rank (r): {self.lora.r}",
            f"  Alpha: {self.lora.alpha}",
            f"  目标模块: {self.lora.target_modules}",
            "",
            "【训练参数】",
            f"  训练轮数: {self.training.num_epochs}",
            f"  批次大小: {self.training.per_device_batch_size}",
            f"  梯度累积: {self.training.gradient_accumulation_steps}",
            f"  学习率: {self.training.learning_rate}",
            f"  精度: {'BF16' if self.training.bf16 else 'FP16'}",
            "",
            "【分布式设置】",
            f"  模式: {self.distributed.mode}",
            f"  GPU数量: {self.distributed.num_gpus}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


def load_model_config(model_name: str, validate: bool = True) -> TrainingConfigLoader:
    loader = TrainingConfigLoader()
    loader.load_by_name(model_name)

    if validate:
        loader.validate(strict=True)

    return loader


def get_config_for_notebook(model_name: str = "gemma4_e4b") -> Dict[str, Any]:
    loader = load_model_config(model_name)
    return loader.to_dict()


def print_config_summary(model_name: str) -> None:
    loader = load_model_config(model_name)
    print(loader.summary())


def list_available_configs() -> List[str]:
    return TrainingConfigLoader.get_available_configs()


def get_lora_adapter_path(
    config: TrainingConfigLoader,
    project_root: Optional[Union[str, Path]] = None,
    distributed_mode: Optional[str] = None,
    timestamp: Optional[str] = None,
    strict: bool = False,
) -> str:
    """
    动态生成 LoRA adapter 的完整路径

    Args:
        config: 已加载的配置对象
        project_root: 项目根目录（可选，默认使用当前工作目录）
        distributed_mode: 分布式训练模式（可选，默认从配置读取）
            - "ddp_{num_gpus}gpu" 格式，如 "ddp_8gpu"
            - "single_gpu"
        timestamp: 时间戳目录名（可选，默认自动获取）
        strict: 是否严格模式，路径不存在时抛出异常

    Returns:
        LoRA adapter 的完整路径字符串

    Raises:
        FileNotFoundError: strict=True 且路径不存在时
        ValueError: 配置缺失或无效时
    """
    import logging

    logger = logging.getLogger(__name__)

    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    base_dir = config.output.base_dir
    lora_subdir = config.output.lora_adapter_subdir

    if not lora_subdir:
        lora_subdir = f"{config.model.name}_lora"
        logger.debug(f"LoRA adapter subdir 未配置，使用默认值: {lora_subdir}")

    if distributed_mode is None:
        dist_cfg = config.distributed
        if dist_cfg.mode == "ddp":
            distributed_mode = f"ddp_{dist_cfg.num_gpus}gpu"
        elif dist_cfg.mode == "single":
            distributed_mode = "single_gpu"
        else:
            distributed_mode = dist_cfg.mode

    if timestamp is None:
        timestamp = config.output.lora_adapter_timestamp
        use_latest = config.output.lora_adapter_use_latest

        if use_latest and not timestamp:
            latest_file = project_root / base_dir / lora_subdir / distributed_mode / "latest.txt"

            if latest_file.exists():
                try:
                    with open(latest_file, "r", encoding="utf-8") as f:
                        timestamp = f.read().strip()
                    logger.info(f"从 latest.txt 读取时间戳: {timestamp}")
                except Exception as e:
                    logger.warning(f"读取 latest.txt 失败: {e}")
                    if strict:
                        raise FileNotFoundError(f"无法读取 latest.txt: {latest_file}")
            else:
                logger.warning(f"latest.txt 不存在: {latest_file}")
                if strict:
                    raise FileNotFoundError(f"latest.txt 不存在: {latest_file}")

    if not timestamp:
        msg = f"未找到有效的时间戳 (lora_adapter_timestamp 或 latest.txt)"
        logger.warning(msg)
        if strict:
            raise ValueError(msg)
        timestamp = ""

    lora_path_parts = [base_dir, lora_subdir, distributed_mode]
    if timestamp:
        lora_path_parts.append(timestamp)

    lora_adapter_path = project_root / Path(*lora_path_parts)

    if strict and not lora_adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter 路径不存在: {lora_adapter_path}")

    logger.info(f"生成 LoRA adapter 路径: {lora_adapter_path}")

    return str(lora_adapter_path)
