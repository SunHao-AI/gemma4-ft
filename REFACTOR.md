# 项目重构说明

## 第一阶段重构：模块拆分与规范化

### 1. 新增共享基础设施模块 `progress_logger.py`

提取各模块中重复的日志配置和进度条创建逻辑，统一到 `progress_logger.py` 模块：

| 功能 | 说明 |
|------|------|
| `setup_progress_logging()` | 配置日志系统，实现控制台输出与日志文件分离 |
| `create_progress_bar()` | 创建标准化的 tqdm 进度条 |
| `PhaseProgressManager` | 多阶段进度管理器 |
| `print_phase_header/footer()` | 阶段信息打印工具 |
| `TQDM_AVAILABLE` | tqdm 库可用性标志 |
| `SUPPORTED_IMAGE_EXTENSIONS` | 支持的图片扩展名集合 |

**tqdm 环境适配**：自动检测运行环境，Jupyter Notebook 中使用 `tqdm.notebook`，普通 Python 中使用标准 `tqdm`。

### 2. labelme_cleaner.py 模块拆分

原 `labelme_cleaner.py` 约 2199 行，包含三个独立功能模块，已拆分为三个文件：

| 新模块 | 内容 | 行数 |
|--------|------|------|
| `labelme_cleaner.py` | ValidationStatus, ValidationResult, CleaningResult, LabelMeCleaner, clean_labelme_data | ~1381 |
| `labelme_statistics.py` | LabelStatistics, LabelMeLabelStatistics, statistics_labelme_labels | ~497 |
| `labelme_stats_processor.py` | FilterCopyResult, StatisticsFileProcessor, process_statistics_file | ~365 |

**命名规范化**：`labelme_filter_copy.py` → `labelme_stats_processor.py`
- 原名"filter_copy"是复合动词短语，不符合 `labelme_<名词>` 命名规范
- 新名与主类 `StatisticsFileProcessor` 名声呼应，更准确描述功能本质

**导入兼容性**：所有公开名称可通过 `from labelme_tools import ...` 访问，__init__.py 已更新导出。

### 3. 日志系统重构：控制台与日志文件分离

| use_tqdm | 控制台行为 | 日志文件行为 |
|----------|-----------|-------------|
| `True` | 仅显示 tqdm 进度条 | 写入全部运行信息 |
| `False` | 显示完整日志 | 写入全部运行信息 |
| `True` + 无 log_file | 仅显示 tqdm 进度条 | NullHandler 兜底 |

### 4. 各模块变更摘要

#### labelme_cleaner.py (拆分后)

- 保留：ValidationStatus, ValidationResult, CleaningResult, LabelMeCleaner, clean_labelme_data, main
- 移除：LabelStatistics, LabelMeLabelStatistics, FilterCopyResult, StatisticsFileProcessor 及相关函数
- 导入优化：移除不再使用的 `create_progress_bar`, `json_loads`, `json_dumps_str`, `ORJSON_AVAILABLE`, `SUPPORTED_IMAGE_EXTENSIONS`

#### labelme_statistics.py (新文件)

- 从 `labelme_cleaner.py` 中提取的统计功能
- 导入：`from .progress_logger import ...`, `from .file_utils import ...`
- 独立 CLI 入口：`statistics_main()`

#### labelme_stats_processor.py (新文件，原 labelme_filter_copy.py)

- 从 `labelme_cleaner.py` 中提取的筛选复制功能
- 导入：`from .progress_logger import ...`, `from .file_utils import ...`
- 独立 CLI 入口：`process_main()`

#### labelme_converter.py

- 导入从 `progress_logger` 和 `file_utils` 统一导入

#### labelme_sampler.py

- 导入从 `progress_logger` 和 `file_utils` 统一导入

#### unzip_tools.py

- `main()` 函数改为 argparse CLI，移除硬编码服务器路径 (`/raid5/sh/data/...`)
- 支持 `--workers`, `--log` 等参数

### 5. color_contrast_tools 清理

- 删除5个冗余迭代版本：`analyzer.py`, `analyzer_v2.py`, `improved_v2.py`, `validation.py`
- 保留核心库 `color_utils.py` + 最终版 `color_contrast_final.py`
- 创建 `__init__.py` 使其成为规范 Python 包

## 第二阶段重构：模块抽离与目录结构重组

### 6. 分布式训练模块抽离

将 `gemma4_multimodal_demo/` 中的分布式训练相关文件迁移到独立的 `distributed_training/` 包（与 `labelme_tools/` 同级）：

| 迁移文件 | 新位置 | 修改内容 |
|----------|--------|---------|
| `distributed_config.py` | `distributed_training/` | 无修改 |
| `train_distributed.py` | `distributed_training/` | 3处导入修改 + 删除 sys.path hack |
| `gpu_monitor.py` | `distributed_training/` | 无修改 |
| `dataset.py` | `distributed_training/` | labelme_tools.progress_logger 导入 |
| `run_distributed.sh` | `distributed_training/` | 包名引用更新 |
| `fsdp_config.json` | `distributed_training/` | 无修改 |
| `requirements.txt` | `distributed_training/` | 无修改 |
| `README.md` | `distributed_training/` | 无修改 |
| `DISTRIBUTED_CONFIG_README.md` | `distributed_training/` | 无修改 |

**`train_distributed.py` 导入修改**：所有内部导入改为 `try/except` 模式，优先使用包名导入 `from distributed_training.xxx`，fallback 到直接导入 `from xxx`（支持 torchrun 直接运行）。

**已删除兼容层**：`gemma4_multimodal_demo/` 整个目录已删除，无代码引用兼容层。

### 7. Jupyter Notebook 迁移

将所有 Notebook 从 `gemma4_multimodal_demo/notebooks/` 移动到项目根目录 `notebooks/`（与 `labelme_tools/` 同级）：

| Notebook | 路径变更 |
|----------|---------|
| `01-data_preparation-labelme_processing.ipynb` | `gemma4_multimodal_demo/notebooks/` → `notebooks/` |
| `02-model_finetuning.ipynb` | `gemma4_multimodal_demo/notebooks/` → `notebooks/` (含4处引用更新) |
| `03-object_detection_demo.ipynb` | `gemma4_multimodal_demo/notebooks/` → `notebooks/` |
| `04-model_comparison.ipynb` | `gemma4_multimodal_demo/notebooks/` → `notebooks/` |
| `Gemma4_(E4B)_Vision.ipynb` | `gemma4_multimodal_demo/notebooks/` → `notebooks/` |

### 8. 文件清理

- 删除 `color_contrast_tools/` 中5个冗余迭代版本文件
- 删除根目录和 `color_contrast_tools/` 下的 `color_contrast_results.json`（debug产物）
- 删除 `test_labelme_filter_copy.py`（已重命名为 `test_labelme_stats_processor.py`）
- 删除 `main.py`（无效占位文件，无代码引用）
- 删除 `gemma4_multimodal_demo/` 整个目录（兼容层已无引用）
- 清理各目录的 `__pycache__/`

### 9. dataset.py tqdm 统一

- 5处 `tqdm()` 直接调用改为 `create_progress_bar()`
- 移除冗余的 `import sys` 和独立 tqdm 导入逻辑
- 通过 `try/except` 优先从 `labelme_tools.progress_logger` 导入，fallback 到原生 tqdm

### 10. pyproject.toml 全面更新

- 添加项目描述和核心依赖列表
- 添加5个可选依赖组：`finetune`, `data`, `monitor`, `dev`
- 添加 ruff lint 和 pytest 配置

## 第三阶段重构：目录命名规范化

### 11. tools → labelme_tools 重命名

将 `tools/` 目录重命名为 `labelme_tools/`，使其名称准确反映功能（LabelMe标注数据处理）：

| 变更类型 | 文件 | 修改内容 |
|----------|------|---------|
| 目录重命名 | `tools/` → `labelme_tools/` | PowerShell Rename-Item |
| 导入更新 | `tests/test_*.py` (7个文件) | `from tools.xxx` → `from labelme_tools.xxx` |
| 导入更新 | `tests/test_progress_logger.py` | `mock.patch("tools.xxx")` → `mock.patch("labelme_tools.xxx")` |
| 导入更新 | `distributed_training/dataset.py` | `from tools.progress_logger` → `from labelme_tools.progress_logger` |
| 文档更新 | `ARCHITECTURE.md` | 全部 `tools/` → `labelme_tools/`，移除已删除项 |
| 文档更新 | `REFACTOR.md` | 全部 `tools` → `labelme_tools` |
| 文档新增 | `labelme_tools/README.md` | 模块文档（流水线/依赖/API/CLI） |

**包内部不受影响**：所有相对导入 (`from .xxx`) 不受包名变更影响。

## 使用方法

### 数据处理工具（labelme_tools 包）

```python
from labelme_tools import clean_labelme_data

result = clean_labelme_data(
    source_dir="path/to/source",
    target_dir="path/to/target",
    log_file="path/to/log.txt",
)
```

```python
from labelme_tools import statistics_labelme_labels

stats = statistics_labelme_labels(
    source_dir="path/to/source",
    output_file="path/to/statistics.json",
)
```

```python
from labelme_tools import process_statistics_file

result = process_statistics_file(
    statistics_file="path/to/statistics.json",
    target_dir="path/to/target",
)
```

### 分布式训练（distributed_training 包）

```python
from distributed_training import DistributedConfig, auto_detect_config

config = auto_detect_config()
print(config.summary())
```

```python
from distributed_training import MultimodalDataset

dataset = MultimodalDataset(
    data_path="path/to/data.jsonl",
    image_load_mode="lazy",
    show_progress=True,
)
```