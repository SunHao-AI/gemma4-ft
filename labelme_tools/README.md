# labelme_tools

LabelMe 标注数据全流程处理工具包，涵盖从原始压缩文件到 Unsloth 微调格式的完整数据流水线。

## 模块概览

| 模块 | 功能 | 主类 | 便捷函数 |
|------|------|------|----------|
| `progress_logger.py` | 进度条与日志基础设施 | `PhaseProgressManager` | `setup_progress_logging`, `create_progress_bar` |
| `file_utils.py` | JSON/文件操作与跨平台链接 | — | `find_json_files`, `parse_json_file`, `write_json_file` |
| `labelme_cleaner.py` | 标注数据清洗与验证 | `LabelMeCleaner` | `clean_labelme_data` |
| `labelme_statistics.py` | 类别统计报告生成 | `LabelMeLabelStatistics` | `statistics_labelme_labels` |
| `labelme_stats_processor.py` | 按统计结果筛选复制 | `StatisticsFileProcessor` | `process_statistics_file` |
| `labelme_sampler.py` | 平衡采样选择 | `LabelMeSampler` | `select_balanced_samples` |
| `labelme_converter.py` | Unsloth 格式转换 | `LabelMeConverter` | `convert_to_unsloth_format` |
| `unzip_tools.py` | 多格式压缩文件解压 | `UnzipTool` | `unzip_files` |

## 模块依赖关系

```
progress_logger ───────────┐
    (基础设施)              │
                           ├── labelme_cleaner
file_utils ────────────────┤── labelme_statistics
    (文件操作)              │── labelme_stats_processor
                           ├── labelme_sampler
                           ├── labelme_converter
                           └── unzip_tools
```

`progress_logger` 和 `file_utils` 为基础层，被所有业务模块依赖；业务模块之间无交叉依赖。

## 数据处理流水线

```
原始压缩文件 ──→ unzip_tools ──→ 解压后 JSON + 图片
                                     │
                     ┌───────────────┤
                     │               │
               labelme_cleaner   labelme_statistics
               (清洗 + 验证)      (类别统计)
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
               (转 Unsloth 格式)            │
                     │                      │
                     └─→ Unsloth 训练数据 ───┘
```

典型使用顺序：解压 → 清洗 → 统计 → 筛选 → 采样 → 转换。

## 快速开始

### 包级导入

所有公开 API 可通过包名直接访问：

```python
from labelme_tools import (
    clean_labelme_data,
    statistics_labelme_labels,
    process_statistics_file,
    select_balanced_samples,
    convert_to_unsloth_format,
    unzip_files,
)
```

### 数据清洗

```python
from labelme_tools import clean_labelme_data

result = clean_labelme_data(
    source_dir="path/to/source",
    target_dir="path/to/target",
    log_file="path/to/log.txt",
)
print(f"清洗完成：{result.valid_files} 有效文件")
```

### 类别统计

```python
from labelme_tools import statistics_labelme_labels

stats = statistics_labelme_labels(
    source_dir="path/to/source",
    output_file="path/to/statistics.json",
)
print(f"发现 {len(stats.label_counts)} 个类别")
```

### 统计结果筛选复制

```python
from labelme_tools import process_statistics_file

result = process_statistics_file(
    statistics_file="path/to/statistics.json",
    target_dir="path/to/target",
)
print(f"复制了 {result.copied_files} 个文件")
```

### 平衡采样

```python
from labelme_tools import select_balanced_samples, SelectionMode

result = select_balanced_samples(
    source_dir="path/to/source",
    target_dir="path/to/target",
    n_images=100,
    mode=SelectionMode.N_IMAGES,
)
print(f"采样了 {len(result.selected_images)} 张图片")
```

### Unsloth 格式转换

```python
from labelme_tools import convert_to_unsloth_format

result = convert_to_unsloth_format(
    source_dir="path/to/source",
    output_dir="path/to/output",
    train_ratio=0.8,
)
print(f"训练集 {len(result.train_records)} 条，验证集 {len(result.val_records)} 条")
```

### 压缩文件解压

```python
from labelme_tools import unzip_files

result = unzip_files(
    source_dir="path/to/archives",
    target_dir="path/to/output",
    workers=4,
)
print(f"解压了 {result.total_files} 个文件")
```

## 性能优化

- **orjson 加速**：`file_utils.py` 优先使用 `orjson`（Rust 实现，3-10x faster）解析 JSON，不可用时自动回退到标准库 `json`
- **并发处理**：清洗、统计、采样、转换等模块均支持 `ThreadPoolExecutor` 并行执行
- **tqdm 适配**：`progress_logger.py` 自动检测运行环境，Jupyter 中使用 `tqdm.notebook`，普通 Python 使用标准 `tqdm`
- **跨平台链接**：`file_utils.py` 的 `create_file_link()` 提供符号链接/硬链接/复制的跨平台兼容实现（Windows 优先硬链接，Linux 优先符号链接）

## CLI 入口

各业务模块内置独立 CLI 入口，支持直接运行：

```bash
python -m labelme_tools.labelme_cleaner --source_dir ./raw --target_dir ./cleaned
python -m labelme_tools.labelme_statistics --source_dir ./cleaned --output_file ./stats.json
python -m labelme_tools.labelme_stats_processor --statistics_file ./stats.json --target_dir ./filtered
python -m labelme_tools.labelme_sampler --source_dir ./filtered --target_dir ./sampled --n_images 100
python -m labelme_tools.labelme_converter --source_dir ./sampled --output_dir ./unsloth_data
python -m labelme_tools.unzip_tools --source_dir ./zips --target_dir ./unzipped
```

## 可选依赖

| 依赖 | 用途 | 影响模块 |
|------|------|----------|
| `orjson` | JSON 高性能解析 | `file_utils` |
| `tqdm` | 进度条显示 | `progress_logger`、所有业务模块 |
| `Pillow` | 图片尺寸获取 | `labelme_sampler`、`labelme_converter` |
| `rarfile` | RAR 格式解压 | `unzip_tools` |
| `py7zr` | 7z 格式解压 | `unzip_tools` |

缺失可选依赖时模块仍可正常运行，相关功能自动降级。