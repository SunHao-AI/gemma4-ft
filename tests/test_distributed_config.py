"""
测试 distributed_config 模块的核心功能
覆盖 DistributedMode, LRScalingStrategy, DeviceMapStrategy, DistributedConfig,
create_ddp_config, create_device_map_config, create_fsdp_config, auto_detect_config
"""

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# Mock 外部依赖模块，避免导入失败
# torch 未安装时需要 mock；PIL/datasets/transformers 等同理
_mock_torch = mock.MagicMock()
_mock_torch_cuda = mock.MagicMock()
_mock_torch_cuda.device_count.return_value = 1
_mock_torch_cuda.is_bf16_supported.return_value = True
_mock_torch_cuda.get_device_properties.return_value = mock.MagicMock(
    total_memory=24 * 1024 ** 3
)
_mock_torch.cuda = _mock_torch_cuda

_mock_modules = {
    "torch": _mock_torch,
    "torch.cuda": _mock_torch_cuda,
    "torch.distributed": mock.MagicMock(),
    "PIL": mock.MagicMock(),
    "PIL.Image": mock.MagicMock(),
    "datasets": mock.MagicMock(),
    "transformers": mock.MagicMock(),
    "transformers.Trainer": mock.MagicMock(),
    "unsloth": mock.MagicMock(),
}
for mod_name, mod_obj in _mock_modules.items():
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mod_obj

from distributed_training.distributed_config import (
    DeviceMapStrategy,
    DistributedConfig,
    DistributedMode,
    LRScalingStrategy,
    auto_detect_config,
    create_ddp_config,
    create_device_map_config,
    create_fsdp_config,
)


def _set_gpu_count(n):
    """设置 mock torch.cuda.device_count 的返回值"""
    _mock_torch_cuda.device_count.return_value = n


def _set_bf16_supported(val):
    """设置 mock torch.cuda.is_bf16_supported 的返回值"""
    _mock_torch_cuda.is_bf16_supported.return_value = val


def _set_device_properties(total_memory):
    """设置 mock torch.cuda.get_device_properties 的返回值"""
    _mock_torch_cuda.get_device_properties.return_value = mock.MagicMock(
        total_memory=total_memory
    )


# ============================================================
# Enum 测试
# ============================================================

class TestDistributedMode:

    def test_all_values(self):
        assert DistributedMode.DDP.value == "ddp"
        assert DistributedMode.DEVICE_MAP.value == "device_map"
        assert DistributedMode.FSDP.value == "fsdp"
        assert DistributedMode.SINGLE_GPU.value == "single_gpu"

    def test_enum_count(self):
        assert len(list(DistributedMode)) == 4


class TestLRScalingStrategy:

    def test_all_values(self):
        assert LRScalingStrategy.NONE.value == "none"
        assert LRScalingStrategy.LINEAR.value == "linear"
        assert LRScalingStrategy.SQRT.value == "sqrt"

    def test_enum_count(self):
        assert len(list(LRScalingStrategy)) == 3


class TestDeviceMapStrategy:

    def test_all_values(self):
        assert DeviceMapStrategy.BALANCED.value == "balanced"
        assert DeviceMapStrategy.AUTO.value == "auto"
        assert DeviceMapStrategy.BALANCED_LOW_0.value == "balanced_low_0"
        assert DeviceMapStrategy.CUSTOM.value == "custom"

    def test_enum_count(self):
        assert len(list(DeviceMapStrategy)) == 4


# ============================================================
# LR缩放测试
# ============================================================

class TestLRScaling:

    def test_none_strategy(self):
        result = DistributedConfig._scale_lr(4e-5, 8, "none")
        assert result == 4e-5

    def test_linear_strategy(self):
        result = DistributedConfig._scale_lr(4e-5, 8, "linear")
        assert result == 4e-5 * 8

    def test_sqrt_strategy(self):
        result = DistributedConfig._scale_lr(4e-5, 8, "sqrt")
        expected = 4e-5 * (8 ** 0.5)
        assert abs(result - expected) < 1e-10

    def test_world_size_1(self):
        result = DistributedConfig._scale_lr(4e-5, 1, "linear")
        assert result == 4e-5

    def test_base_lr_preserved_none(self):
        result = DistributedConfig._scale_lr(1e-4, 16, "none")
        assert result == 1e-4


# ============================================================
# DistributedConfig 验证测试
# ============================================================

class TestDistributedConfigValidation:

    def test_invalid_mode(self):
        _set_gpu_count(1)
        with pytest.raises(ValueError, match="mode必须是"):
            DistributedConfig(mode="invalid_mode")

    def test_models_per_gpu_less_than_1(self):
        _set_gpu_count(1)
        with pytest.raises(ValueError, match="models_per_gpu必须>=1"):
            DistributedConfig(
                mode="single_gpu",
                models_per_gpu=0,
                gpu_ids=[0],
            )

    def test_ddp_with_device_map_strategy(self):
        _set_gpu_count(2)
        with pytest.raises(ValueError, match="DDP模式与device_map互斥"):
            DistributedConfig(
                mode="ddp",
                gpu_ids=[0, 1],
                device_map_strategy="balanced",
            )

    def test_fsdp_with_device_map_strategy(self):
        _set_gpu_count(2)
        with pytest.raises(ValueError, match="FSDP模式与device_map互斥"):
            DistributedConfig(
                mode="fsdp",
                gpu_ids=[0, 1],
                device_map_strategy="balanced",
            )

    def test_device_map_without_strategy_or_custom(self):
        _set_gpu_count(2)
        with pytest.raises(ValueError, match="device_map模式需要指定"):
            DistributedConfig(
                mode="device_map",
                gpu_ids=[0, 1],
            )

    def test_invalid_lr_scaling(self):
        _set_gpu_count(1)
        with pytest.raises(ValueError, match="lr_scaling必须是"):
            DistributedConfig(mode="single_gpu", gpu_ids=[0], lr_scaling="invalid")

    def test_invalid_image_load_mode(self):
        _set_gpu_count(1)
        with pytest.raises(ValueError, match="image_load_mode必须是"):
            DistributedConfig(
                mode="single_gpu",
                gpu_ids=[0],
                image_load_mode="invalid",
            )

    def test_gpu_groups_overlap(self):
        _set_gpu_count(4)
        with pytest.raises(ValueError, match="重叠GPU"):
            DistributedConfig(
                mode="device_map",
                gpu_ids=[0, 1],
                gpu_groups=[[0, 2], [1, 3]],
                device_map_strategy="balanced",
            )

    def test_empty_gpu_group(self):
        _set_gpu_count(4)
        with pytest.raises(ValueError, match="GPU分组不能为空"):
            DistributedConfig(
                mode="device_map",
                gpu_groups=[[0, 1], []],
                device_map_strategy="balanced",
            )

    def test_models_per_gpu_auto_set_to_1_in_device_map(self):
        _set_gpu_count(4)
        config = DistributedConfig(
            mode="device_map",
            gpu_groups=[[0, 1], [2, 3]],
            device_map_strategy="balanced",
            models_per_gpu=2,
        )
        assert config.models_per_gpu == 1


# ============================================================
# DistributedConfig 属性测试
# ============================================================

class TestDistributedConfigProperties:

    def test_single_gpu_config(self):
        _set_gpu_count(1)
        config = DistributedConfig(mode="single_gpu", gpu_ids=[0])
        assert config.mode == "single_gpu"
        assert config.num_data_parallel_groups == 1
        assert config.gpus_per_model == 1
        assert config.total_parallel_backward == 1

    def test_ddp_config_effective_batch(self):
        _set_gpu_count(2)
        config = DistributedConfig(
            mode="ddp",
            gpu_ids=[0, 1],
            per_device_batch_size=4,
            gradient_accumulation_steps=2,
            models_per_gpu=2,
        )
        assert config.effective_global_batch == 32

    def test_ddp_config_effective_lr_linear(self):
        _set_gpu_count(4)
        config = DistributedConfig(
            mode="ddp",
            gpu_ids=[0, 1, 2, 3],
            learning_rate=4e-5,
            lr_scaling="linear",
        )
        assert config.effective_lr == 4e-5 * 4

    def test_ddp_config_effective_lr_sqrt(self):
        _set_gpu_count(4)
        config = DistributedConfig(
            mode="ddp",
            gpu_ids=[0, 1, 2, 3],
            learning_rate=4e-5,
            lr_scaling="sqrt",
        )
        expected = 4e-5 * (4 ** 0.5)
        assert abs(config.effective_lr - expected) < 1e-10

    def test_ddp_config_no_lr_scaling(self):
        _set_gpu_count(4)
        config = DistributedConfig(
            mode="ddp",
            gpu_ids=[0, 1, 2, 3],
            learning_rate=4e-5,
            lr_scaling="none",
        )
        assert config.effective_lr == 4e-5

    def test_device_map_config_parallel_groups(self):
        _set_gpu_count(4)
        config = DistributedConfig(
            mode="device_map",
            gpu_groups=[[0, 1], [2, 3]],
            device_map_strategy="balanced",
        )
        assert config.num_data_parallel_groups == 2
        assert config.gpus_per_model == 2

    def test_fsdp_config(self):
        _set_gpu_count(4)
        config = DistributedConfig(
            mode="fsdp",
            gpu_ids=[0, 1, 2, 3],
        )
        assert config.num_data_parallel_groups == 4
        assert config.gpus_per_model == 1


# ============================================================
# DistributedConfig device_map 测试
# ============================================================

class TestDistributedConfigDeviceMap:

    def test_single_gpu_device_map(self):
        _set_gpu_count(1)
        config = DistributedConfig(mode="single_gpu", gpu_ids=[0])
        device_map = config.get_device_map()
        assert device_map is not None
        assert device_map[""] == 0

    def test_ddp_device_map_non_distributed(self):
        _set_gpu_count(2)
        config = DistributedConfig(mode="ddp", gpu_ids=[0, 1])
        with mock.patch.dict(os.environ, {}, clear=True):
            device_map = config.get_device_map()
        assert device_map is not None
        assert device_map[""] == 0

    def test_ddp_device_map_distributed_env(self):
        _set_gpu_count(2)
        config = DistributedConfig(mode="ddp", gpu_ids=[0, 1])
        with mock.patch.dict(os.environ, {"LOCAL_RANK": "0"}, clear=False):
            device_map = config.get_device_map()
        assert device_map is None

    def test_device_map_with_custom_map(self):
        _set_gpu_count(2)
        config = DistributedConfig(
            mode="device_map",
            custom_device_map={"model.embed_tokens": 0, "model.layers": 1},
            device_map_strategy="custom",
        )
        device_map = config.get_device_map()
        assert device_map == {"model.embed_tokens": 0, "model.layers": 1}

    def test_device_map_with_strategy(self):
        _set_gpu_count(2)
        config = DistributedConfig(
            mode="device_map",
            gpu_groups=[[0, 1]],
            device_map_strategy="auto",
        )
        device_map = config.get_device_map()
        assert device_map is not None

    def test_fsdp_device_map_is_none(self):
        _set_gpu_count(2)
        config = DistributedConfig(mode="fsdp", gpu_ids=[0, 1])
        assert config.get_device_map() is None


# ============================================================
# DistributedConfig CUDA_VISIBLE_DEVICES 测试
# ============================================================

class TestDistributedConfigCUDAVisible:

    def test_gpu_ids(self):
        _set_gpu_count(8)
        config = DistributedConfig(mode="ddp", gpu_ids=[0, 2, 4, 6])
        assert config.get_cuda_visible_devices() == "0,2,4,6"

    def test_gpu_groups(self):
        _set_gpu_count(4)
        config = DistributedConfig(
            mode="device_map",
            gpu_groups=[[0, 1], [2, 3]],
            device_map_strategy="balanced",
        )
        assert config.get_cuda_visible_devices() == "0,1,2,3"

    def test_no_gpu_ids_no_groups(self):
        _set_gpu_count(4)
        config = DistributedConfig(mode="ddp")
        assert config.get_cuda_visible_devices() == "0,1,2,3"


# ============================================================
# DistributedConfig 序列化测试
# ============================================================

class TestDistributedConfigSerialization:

    def test_to_dict(self):
        _set_gpu_count(1)
        config = DistributedConfig(mode="single_gpu", gpu_ids=[0])
        d = config.to_dict()
        assert d["mode"] == "single_gpu"
        assert d["gpu_ids"] == [0]
        assert "effective_global_batch" in d
        assert "effective_lr" in d
        assert "total_parallel_backward" in d

    def test_from_dict(self):
        _set_gpu_count(1)
        d = {"mode": "single_gpu", "gpu_ids": [0], "learning_rate": 5e-5}
        config = DistributedConfig.from_dict(d)
        assert config.mode == "single_gpu"
        assert config.learning_rate == 5e-5

    def test_from_dict_unknown_fields_filtered(self):
        _set_gpu_count(1)
        d = {"mode": "single_gpu", "gpu_ids": [0], "unknown_field": "value"}
        config = DistributedConfig.from_dict(d)
        assert config.mode == "single_gpu"

    def test_to_json_and_from_json(self, tmp_path):
        _set_gpu_count(1)
        config = DistributedConfig(
            mode="single_gpu",
            gpu_ids=[0],
            learning_rate=3e-5,
        )
        json_path = str(tmp_path / "config.json")
        config.to_json(json_path)

        _set_gpu_count(1)
        loaded = DistributedConfig.from_json(json_path)
        assert loaded.mode == "single_gpu"
        assert loaded.learning_rate == 3e-5


# ============================================================
# DistributedConfig summary 测试
# ============================================================

class TestDistributedConfigSummary:

    def test_summary_contains_mode(self):
        _set_gpu_count(1)
        config = DistributedConfig(mode="single_gpu", gpu_ids=[0])
        summary = config.summary()
        assert "single_gpu" in summary
        assert "分布式训练配置摘要" in summary

    def test_summary_contains_training_params(self):
        _set_gpu_count(1)
        config = DistributedConfig(
            mode="single_gpu",
            gpu_ids=[0],
            per_device_batch_size=8,
            learning_rate=2e-5,
        )
        summary = config.summary()
        assert "每GPU批次" in summary
        assert "学习率" in summary


# ============================================================
# DistributedConfig training_kwargs 测试
# ============================================================

class TestDistributedConfigTrainingKwargs:

    def test_basic_kwargs(self):
        _set_gpu_count(1)
        config = DistributedConfig(mode="single_gpu", gpu_ids=[0])
        _set_bf16_supported(True)
        kwargs = config.get_training_kwargs()
        assert "per_device_train_batch_size" in kwargs
        assert "learning_rate" in kwargs
        assert "num_train_epochs" in kwargs
        assert "bf16" in kwargs

    def test_ddp_kwargs(self):
        _set_gpu_count(2)
        config = DistributedConfig(mode="ddp", gpu_ids=[0, 1])
        _set_bf16_supported(True)
        kwargs = config.get_training_kwargs()
        assert kwargs["ddp_find_unused_parameters"] is False

    def test_dataloader_kwargs(self):
        _set_gpu_count(1)
        config = DistributedConfig(
            mode="single_gpu",
            gpu_ids=[0],
            dataloader_num_workers=6,
            dataloader_prefetch_factor=4,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
        )
        _set_bf16_supported(True)
        kwargs = config.get_training_kwargs()
        assert kwargs["dataloader_num_workers"] == 6
        assert kwargs["dataloader_prefetch_factor"] == 4
        assert kwargs["dataloader_pin_memory"] is True
        assert kwargs["dataloader_persistent_workers"] is True

    def test_vision_mode_kwargs(self):
        _set_gpu_count(1)
        config = DistributedConfig(
            mode="single_gpu",
            gpu_ids=[0],
            vision_mode=True,
        )
        _set_bf16_supported(True)
        kwargs = config.get_training_kwargs()
        assert kwargs["remove_unused_columns"] is False
        assert kwargs["dataset_text_field"] == ""


# ============================================================
# DistributedConfig model_kwargs 测试
# ============================================================

class TestDistributedConfigModelKwargs:

    def test_model_kwargs_load_in_4bit(self):
        _set_gpu_count(1)
        config = DistributedConfig(
            mode="single_gpu",
            gpu_ids=[0],
            load_in_4bit=True,
            model_name="test_model",
        )
        kwargs = config.get_model_kwargs()
        assert kwargs["load_in_4bit"] is True
        assert kwargs["model_name"] == "test_model"

    def test_model_kwargs_vision_mode(self):
        _set_gpu_count(1)
        config = DistributedConfig(
            mode="single_gpu",
            gpu_ids=[0],
            vision_mode=True,
        )
        kwargs = config.get_model_kwargs()
        assert kwargs["disable_log_stats"] is True


# ============================================================
# DistributedConfig FSDP 配置加载测试
# ============================================================

class TestDistributedConfigFSDPConfig:

    def test_default_fsdp_config(self):
        _set_gpu_count(2)
        config = DistributedConfig(mode="fsdp", gpu_ids=[0, 1])
        fsdp_cfg = config._load_fsdp_config()
        assert "fsdp_auto_wrap_policy" in fsdp_cfg
        assert fsdp_cfg["fsdp_auto_wrap_policy"] == "TRANSFORMER_BASED_WRAP"

    def test_custom_fsdp_config(self):
        custom_cfg = {
            "fsdp_auto_wrap_policy": "SIZE_BASED_WRAP",
            "fsdp_sharding_strategy": "SHARD_GRAD_OP",
        }
        _set_gpu_count(2)
        config = DistributedConfig(
            mode="fsdp",
            gpu_ids=[0, 1],
            fsdp_config=custom_cfg,
        )
        fsdp_cfg = config._load_fsdp_config()
        assert fsdp_cfg["fsdp_auto_wrap_policy"] == "SIZE_BASED_WRAP"

    def test_fsdp_config_from_file(self, tmp_path):
        cfg_data = {
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "custom_key": "custom_value",
        }
        cfg_file = tmp_path / "fsdp_config.json"
        cfg_file.write_text(json.dumps(cfg_data), encoding="utf-8")

        _set_gpu_count(2)
        config = DistributedConfig(
            mode="fsdp",
            gpu_ids=[0, 1],
            fsdp_config_path=str(cfg_file),
        )
        fsdp_cfg = config._load_fsdp_config()
        assert fsdp_cfg["custom_key"] == "custom_value"


# ============================================================
# DistributedConfig torchrun 命令测试
# ============================================================

class TestDistributedConfigTorchrunCommand:

    def test_ddp_torchrun_command(self):
        _set_gpu_count(2)
        config = DistributedConfig(mode="ddp", gpu_ids=[0, 1])
        cmd = config.get_torchrun_command()
        assert "torchrun" in cmd
        assert "--nproc_per_node=2" in cmd

    def test_single_gpu_command(self):
        _set_gpu_count(1)
        config = DistributedConfig(mode="single_gpu", gpu_ids=[0])
        cmd = config.get_torchrun_command()
        assert "python" in cmd
        assert "torchrun" not in cmd

    def test_fsdp_torchrun_command(self):
        _set_gpu_count(2)
        config = DistributedConfig(mode="fsdp", gpu_ids=[0, 1])
        cmd = config.get_torchrun_command()
        assert "--use_fsdp" in cmd

    def test_device_map_torchrun_command(self):
        _set_gpu_count(4)
        config = DistributedConfig(
            mode="device_map",
            gpu_groups=[[0, 1], [2, 3]],
            device_map_strategy="balanced",
        )
        cmd = config.get_torchrun_command()
        assert "--use_ddp" in cmd
        assert "--device_map" in cmd

    def test_command_contains_runtime_tuning_args(self):
        _set_gpu_count(2)
        config = DistributedConfig(
            mode="ddp",
            gpu_ids=[0, 1],
            dataloader_num_workers=8,
            dataloader_prefetch_factor=4,
            cpu_threads_per_rank=6,
            tf32=True,
            image_load_mode="lazy",
        )
        cmd = config.get_torchrun_command()
        assert "--dataloader_num_workers 8" in cmd
        assert "--dataloader_prefetch_factor 4" in cmd
        assert "--cpu_threads_per_rank 6" in cmd
        assert "--tf32" in cmd


# ============================================================
# 便捷函数测试
# ============================================================

class TestCreateDDPConfig:

    def test_basic_creation(self):
        _set_gpu_count(2)
        config = create_ddp_config(gpu_ids=[0, 1])
        assert config.mode == "ddp"

    def test_with_models_per_gpu(self):
        _set_gpu_count(2)
        config = create_ddp_config(
            gpu_ids=[0, 1],
            models_per_gpu=2,
        )
        assert config.models_per_gpu == 2
        assert config.effective_global_batch > config.per_device_batch_size


class TestCreateDeviceMapConfig:

    def test_basic_creation(self):
        _set_gpu_count(4)
        config = create_device_map_config(
            gpu_groups=[[0, 1], [2, 3]],
        )
        assert config.mode == "device_map"
        assert config.num_data_parallel_groups == 2

    def test_with_custom_device_map(self):
        custom_map = {"model.embed_tokens": 0}
        _set_gpu_count(2)
        config = create_device_map_config(
            custom_device_map=custom_map,
            device_map_strategy="custom",
        )
        assert config.custom_device_map == custom_map


class TestCreateFSDPConfig:

    def test_basic_creation(self):
        _set_gpu_count(4)
        config = create_fsdp_config(gpu_ids=[0, 1, 2, 3])
        assert config.mode == "fsdp"

    def test_with_custom_fsdp_config(self):
        fsdp_cfg = {"fsdp_sharding_strategy": "SHARD_GRAD_OP"}
        _set_gpu_count(2)
        config = create_fsdp_config(
            gpu_ids=[0, 1],
            fsdp_config=fsdp_cfg,
        )
        assert config.fsdp_config == fsdp_cfg


class TestAutoDetectConfig:

    def test_small_model_ddp(self):
        _set_gpu_count(2)
        _set_device_properties(24 * 1024 ** 3)
        config = auto_detect_config(model_vram_gb=5.0)
        assert config.mode == "ddp"

    def test_large_model_device_map(self):
        _set_gpu_count(4)
        _set_device_properties(24 * 1024 ** 3)
        config = auto_detect_config(model_vram_gb=20.0)
        assert config.mode == "device_map"

    def test_huge_model_fsdp(self):
        _set_gpu_count(2)
        _set_device_properties(24 * 1024 ** 3)
        config = auto_detect_config(model_vram_gb=100.0)
        assert config.mode == "fsdp"

    def test_no_gpu_raises_error(self):
        _set_gpu_count(0)
        with pytest.raises(RuntimeError, match="未检测到GPU"):
            auto_detect_config()
