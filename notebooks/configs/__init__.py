"""
模型配置管理系统

提供统一的配置文件加载、验证和访问接口。

使用方式:
    # 方式1: 按模型名称加载
    from notebooks.configs import load_model_config
    config = load_model_config("gemma4_e4b")
    
    # 方式2: 直接获取配置字典
    from notebooks.configs import get_config_for_notebook
    config_dict = get_config_for_notebook("gemma4_e4b")
    
    # 方式3: 列出可用配置
    from notebooks.configs import list_available_configs
    print(list_available_configs())

切换模型:
    只需更改加载的配置名称即可:
    config = load_model_config("qwen3_5_4b")  # 切换到 Qwen3.5
"""

from .config_loader import (
    ConfigValidationError,
    DataLoaderConfig,
    DataPreparationConfig,
    DistributedConfig,
    EvaluationConfig,
    InferenceConfig,
    LoRAConfig,
    ModelConfig,
    ModelLoadingConfig,
    OutputConfig,
    TrainingConfig,
    TrainingConfigLoader,
    get_config_for_notebook,
    list_available_configs,
    load_model_config,
    print_config_summary,
)

__all__ = [
    "ConfigValidationError",
    "DataLoaderConfig",
    "DataPreparationConfig",
    "DistributedConfig",
    "EvaluationConfig",
    "InferenceConfig",
    "LoRAConfig",
    "ModelConfig",
    "ModelLoadingConfig",
    "OutputConfig",
    "TrainingConfig",
    "TrainingConfigLoader",
    "get_config_for_notebook",
    "list_available_configs",
    "load_model_config",
    "print_config_summary",
]