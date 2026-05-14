# gemma4-ft 项目架构文档

## 项目概述

gemma4-ft 是 Gemma 4 多模态微调项目，包含分布式训练工具包与数据处理流水线。项目按功能划分为三大模块包：

```
gemma4-ft/
├── labelme_tools/               # LabelMe标注数据处理工具包
│   ├── progress_logger.py       # 进度日志基础设施
│   ├── file_utils.py            # 文件操作工具
│   ├── labelme_cleaner.py       # LabelMe数据清洗
│   ├── labelme_statistics.py    # LabelMe类别统计
│   ├── labelme_stats_processor.py  # 统计结果筛选复制
│   ├── labelme_sampler.py       # 平衡采样选择
│   ├── labelme_converter.py     # Unsloth格式转换
│   ├── unzip_tools.py           # 压缩文件解压
│   ├── README.md                # 模块文档
│   └── __init__.py              # 包入口（统一导出）
│
├── distributed_training/        # 分布式训练工具包
│   ├── distributed_config.py    # 分布式配置管理
│   ├── train_distributed.py     # 训练脚本入口
│   ├── gpu_monitor.py           # GPU监控工具
│   ├── dataset.py               # 多模态数据集
│   ├── fsdp_config.json         # FSDP配置文件
│   ├── requirements.txt         # 训练依赖清单
│   ├── run_distributed.sh       # torchrun启动脚本
│   ├── README.md                # 训练模块文档
│   ├── DISTRIBUTED_CONFIG_README.md  # 配置模块文档
│   └── __init__.py              # 包入口（统一导出）
│
├── color_contrast_tools/        # WCAG颜色对比度工具
│   ├── color_utils.py           # 颜色计算核心函数
│   ├── color_contrast_final.py  # 最终方案验证脚本
│   ├── README.md                # 模块文档
│   ├── COLOR_STYLE_GUIDE.md     # 颜色规范文档
│   └── __init__.py              # 包入口
│
├── notebooks/                   # Jupyter Notebook
│   ├── 01-data_preparation-labelme_processing.ipynb
│   ├── 02-model_finetuning.ipynb
│   ├── 03-object_detection_demo.ipynb
│   ├── 04-model_comparison.ipynb
│   └── Gemma4_(E4B)_Vision.ipynb
│
├── tests/                       # 单元测试
│   ├── test_progress_logger.py
│   ├── test_file_utils.py
│   ├── test_labelme_cleaner.py
│   ├── test_labelme_statistics.py
│   ├── test_labelme_stats_processor.py
│   ├── test_labelme_sampler.py
│   ├── test_labelme_converter.py
│   ├── test_distributed_config.py
│   ├── test_color_utils.py
│   └── __init__.py
│
├── pyproject.toml               # 项目配置与依赖
└── AGENTS.md                    # Agent行为规范
```

## 模块依赖关系

### labelme_tools 包内部依赖

```
progress_logger ──────────────────┐
    (基础设施)                     │
                                  ├── labelme_cleaner
file_utils ───────────────────────┤── labelme_statistics
    (文件操作)                     │── labelme_stats_processor
                                  ├── labelme_sampler
                                  ├── labelme_converter
                                  └── unzip_tools
```

`progress_logger` 和 `file_utils` 是基础层，被所有业务模块依赖。业务模块之间无交叉依赖。

### distributed_training 包内部依赖

```
distributed_config ──→ train_distributed ──→ dataset
                          │
                          └─→ gpu_monitor
```

`train_distributed.py` 是编排层，依赖配置、数据集和监控三个独立模块。

### 跨包依赖

```
distributed_training/dataset.py ──→ labelme_tools/progress_logger
```

`dataset.py` 使用 `labelme_tools.progress_logger` 的 `TQDM_AVAILABLE` 和 `create_progress_bar()` 统一进度管理。

## 核心数据流水线

LabelMe标注数据的完整处理流水线：

```
原始压缩文件 ──→ unzip_tools ──→ 解压后JSON+图片
                                    │
                    ┌───────────────┤
                    │               │
              labelme_cleaner   labelme_statistics
              (清洗+验证)       (类别统计)
                    │               │
                    │               └─→ statistics.json
                    │                      │
                    │               labelme_stats_processor
                    │               (按统计结果筛选复制)
                    │                      │
              labelme_sampler              │
              (平衡采样)                   │
                    │                      │
              labelme_converter ───────────┤
              (转Unsloth格式)              │
                    │                      │
                    └─→ Unsloth训练数据 ────┘
                           │
                           └─→ distributed_training/MultimodalDataset
                                  │
                                  └─→ 模型微调
```

## 模块API摘要

### labelme_tools.progress_logger

| API | 类型 | 说明 |
|-----|------|------|
| `TQDM_AVAILABLE` | bool | tqdm是否可用 |
| `IN_NOTEBOOK` | bool | 是否在Jupyter环境 |
| `SUPPORTED_IMAGE_EXTENSIONS` | set | 支持的图片扩展名 |
| `setup_progress_logging(name, log_file, use_tqdm)` | func | 配置日志系统 |
| `create_progress_bar(total, desc, unit, **kwargs)` | func | 创建进度条 |
| `PhaseProgressManager(phases, use_tqdm)` | class | 多阶段进度管理 |
| `print_phase_header/footer(phase, ...)` | func | 阶段信息打印 |

### labelme_tools.file_utils

| API | 类型 | 说明 |
|-----|------|------|
| `ORJSON_AVAILABLE` | bool | orjson是否可用 |
| `SUPPORTED_IMAGE_EXTENSIONS` | set | 图片扩展名集合 |
| `find_json_files(dir, recursive)` | func | 查找JSON文件 |
| `parse_json_file(path, encoding)` | func | 解析JSON（自动orjson+编码） |
| `find_image_file(json_path, extensions)` | func | 查找关联图片 |
| `get_relative_path(path, base)` | func | 获取相对路径 |
| `json_loads(s)` | func | JSON字符串解析 |
| `json_dumps_str(obj)` | func | JSON序列化字符串 |
| `write_json_file(path, data, ...)` | func | 写JSON文件 |
| `create_file_link(src, dst, link_type)` | func | 跨平台文件链接 |

### labelme_tools.labelme_cleaner

| API | 类型 | 说明 |
|-----|------|------|
| `ValidationStatus` | enum | 验证状态枚举 |
| `ValidationResult` | dataclass | 单文件验证结果 |
| `CleaningResult` | dataclass | 清洗汇总结果 |
| `LabelMeCleaner` | class | 清洗工具（5阶段流水线） |
| `clean_labelme_data(...)` | func | 便捷清洗函数 |

### labelme_tools.labelme_statistics

| API | 类型 | 说明 |
|-----|------|------|
| `LabelStatistics` | dataclass | 统计结果数据类 |
| `LabelMeLabelStatistics` | class | 统计工具类 |
| `statistics_labelme_labels(...)` | func | 便捷统计函数 |

### labelme_tools.labelme_stats_processor

| API | 类型 | 说明 |
|-----|------|------|
| `FilterCopyResult` | dataclass | 筛选复制结果 |
| `StatisticsFileProcessor` | class | 统计文件处理器 |
| `process_statistics_file(...)` | func | 便捷处理函数 |

### labelme_tools.labelme_sampler

| API | 类型 | 说明 |
|-----|------|------|
| `SelectionMode` | enum | N_IMAGES / N_LABELS |
| `ImageLabelInfo` | dataclass | 图片标签信息 |
| `SelectionResult` | dataclass | 采样结果 |
| `BalancedSelectionResult` | dataclass | 平衡采样结果 |
| `LabelMeSampler` | class | 采样工具类 |
| `select_balanced_samples(...)` | func | 便捷采样函数 |

### labelme_tools.labelme_converter

| API | 类型 | 说明 |
|-----|------|------|
| `BoundingBox` | dataclass | 边界框 |
| `ConversionRecord` | dataclass | 转换记录 |
| `DatasetSplit` | dataclass | 数据集分割结果 |
| `ConversionResult` | dataclass | 转换汇总结果 |
| `LabelMeConverter` | class | 格式转换器 |
| `convert_to_unsloth_format(...)` | func | 便捷转换函数 |

### labelme_tools.unzip_tools

| API | 类型 | 说明 |
|-----|------|------|
| `UnzipResult` | dataclass | 解压结果 |
| `UnzipTool` | class | 解压工具类 |
| `unzip_files(...)` | func | 便捷解压函数 |

### distributed_training.distributed_config

| API | 类型 | 说明 |
|-----|------|------|
| `DistributedMode` | enum | DDP/DEVICE_MAP/FSDP/SINGLE_GPU |
| `LRScalingStrategy` | enum | 学习率缩放策略 |
| `DeviceMapStrategy` | enum | 设备映射策略 |
| `DistributedConfig` | dataclass | 统一配置类（含自动计算字段） |
| `auto_detect_config()` | func | 自动检测最佳配置 |
| `create_ddp/device_map/fsdp_config()` | func | 快捷配置创建 |

### distributed_training.gpu_monitor

| API | 类型 | 说明 |
|-----|------|------|
| `GPUMonitor` | class | GPU监控（CSV日志+nvidia-smi） |
| `GPUMonitorCallback` | class | HuggingFace Trainer回调 |
| `benchmark_single_vs_multi()` | func | 单卡/多卡性能对比 |
| `print_gpu_info()` | func | 打印GPU信息 |

### distributed_training.dataset

| API | 类型 | 说明 |
|-----|------|------|
| `MultimodalDataset` | class | 多模态数据集（lazy/preload/batch加载） |
| `create_multimodal_dataset()` | func | 便捷创建函数 |
| `create_vision_dataset()` | func | 视觉微调数据集创建 |

### color_contrast_tools.color_utils

| API | 类型 | 说明 |
|-----|------|------|
| `hex_to_rgb(hex)` | func | 十六进制转RGB |
| `rgb_to_hex(r,g,b)` | func | RGB转十六进制 |
| `srgb_to_linear(channel)` | func | sRGB转线性亮度 |
| `get_relative_luminance(r,g,b)` | func | 计算相对亮度（WCAG 2.1） |
| `calculate_contrast_ratio(c1,c2)` | func | 计算对比度比率 |
| `wcag_compliance(ratio)` | func | WCAG合规性判断 |