"""distributed_training - 分布式训练工具包"""

from distributed_training.distributed_config import (
    DistributedConfig,
    DistributedMode,
    LRScalingStrategy,
    DeviceMapStrategy,
    auto_detect_config,
    create_ddp_config,
    create_device_map_config,
    create_fsdp_config,
)

from distributed_training.dataset import (
    MultimodalDataset,
    create_multimodal_dataset,
    create_vision_dataset,
)

from distributed_training.gpu_monitor import (
    GPUMonitor,
    GPUMonitorCallback,
    benchmark_single_vs_multi,
    print_gpu_info,
)

__all__ = [
    "DistributedConfig",
    "DistributedMode",
    "LRScalingStrategy",
    "DeviceMapStrategy",
    "auto_detect_config",
    "create_ddp_config",
    "create_device_map_config",
    "create_fsdp_config",
    "MultimodalDataset",
    "create_multimodal_dataset",
    "create_vision_dataset",
    "GPUMonitor",
    "GPUMonitorCallback",
    "benchmark_single_vs_multi",
    "print_gpu_info",
]