"""gemma4_multimodal_demo - Gemma 4 多模态微调演示模块"""

from gemma4_multimodal_demo.dataset import MultimodalDataset, create_multimodal_dataset, create_vision_dataset
from gemma4_multimodal_demo.distributed_config import (
    DistributedConfig,
    DistributedMode,
    LRScalingStrategy,
    DeviceMapStrategy,
    auto_detect_config,
    create_ddp_config,
    create_device_map_config,
    create_fsdp_config,
)

__all__ = [
    "MultimodalDataset",
    "create_multimodal_dataset",
    "create_vision_dataset",
    "DistributedConfig",
    "DistributedMode",
    "LRScalingStrategy",
    "DeviceMapStrategy",
    "auto_detect_config",
    "create_ddp_config",
    "create_device_map_config",
    "create_fsdp_config",
]