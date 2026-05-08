# tools 模块重构说明

## 重构内容

### 1. 新增共享基础设施模块 `progress_logger.py`

提取各模块中重复的日志配置和进度条创建逻辑，统一到 `progress_logger.py` 模块：

| 功能 | 说明 |
|------|------|
| `setup_progress_logging()` | 配置日志系统，实现控制台输出与日志文件分离 |
| `create_progress_bar()` | 创建标准化的 tqdm 进度条 |
| `TQDM_AVAILABLE` | tqdm 库可用性标志 |
| `SUPPORTED_IMAGE_EXTENSIONS` | 支持的图片扩展名集合 |

**tqdm 环境适配**：自动检测运行环境，Jupyter Notebook 中使用 `tqdm.notebook`，普通 Python 中使用标准 `tqdm`：

```python
if hasattr(__builtins__, "__IPYTHON__"):
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange
```

### 2. 日志系统重构：控制台与日志文件分离

**核心设计原则**：

| use_tqdm | 控制台行为 | 日志文件行为 |
|----------|-----------|-------------|
| `True` | 仅显示 tqdm 进度条 | 写入全部运行信息 |
| `False` | 显示完整日志 | 写入全部运行信息 |
| `True` + 无 log_file | 仅显示 tqdm 进度条 | NullHandler 兜底（防止泄漏到 stderr） |

**关键实现细节**：

- `logger.propagate = False`：防止日志消息泄漏到根日志记录器
- `NullHandler` 兜底：use_tqdm=True 且无 log_file 时，确保日志不泄漏到 stderr
- 所有 `self._pbar.write()` 调用替换为 `self.logger.info()` / `self.logger.warning()`，运行信息写入日志文件而非控制台
- 移除 `set_postfix_str()` 调用，进度条保持简洁格式

### 3. 移除重复代码

| 移除项 | 原位置 | 替代方案 |
|--------|--------|---------|
| `TqdmLoggingHandler` 类 | 各模块独立定义 | `setup_progress_logging()` |
| `_setup_logging()` 方法 | 各类中独立定义 | `setup_progress_logging()` |
| `_tqdm_handler` 属性 | 各类中独立定义 | 不再需要 |
| `SUPPORTED_IMAGE_EXTENSIONS` 常量 | 各模块独立定义 | 从 `progress_logger` 导入 |
| `_extract_single_file_with_pbar()` | `UnzipTool` 中 | 合并入 `_extract_single_file()` |

### 4. 各模块变更摘要

#### labelme_cleaner.py

- 导入：移除 `sys`、`TqdmLoggingHandler`、本地 `tqdm` 导入和 `SUPPORTED_IMAGE_EXTENSIONS`，从 `progress_logger` 导入
- `LabelMeCleaner.__init__`：移除 `_tqdm_handler`，用 `setup_progress_logging()` 替代 `_setup_logging()`
- `LabelMeCleaner.clean()`：用 `create_progress_bar()` 替代手动 `tqdm()` 创建
- `LabelMeLabelStatistics.__init__`：新增 `use_tqdm` 参数，用 `setup_progress_logging()` 替代 `_setup_logging()`
- `LabelMeLabelStatistics.statistics()`：新增 tqdm 进度条支持
- `StatisticsFileProcessor.__init__`：新增 `use_tqdm` 参数，用 `setup_progress_logging()` 替代 `_setup_logging()`
- `StatisticsFileProcessor.process()`：新增 tqdm 进度条支持
- 顶层函数 `statistics_labelme_labels()` 和 `process_statistics_file()`：新增 `use_tqdm` 参数

#### labelme_converter.py

- 导入：移除 `sys`、`TqdmLoggingHandler`、本地 `tqdm` 导入和 `SUPPORTED_IMAGE_EXTENSIONS`，从 `progress_logger` 导入
- `LabelMeConverter.__init__`：移除 `_tqdm_handler`，用 `setup_progress_logging()` 替代 `_setup_logging()`
- `LabelMeConverter.convert()`：用 `create_progress_bar()` 替代手动 `tqdm()` 创建
- `LabelMeConverter._save_split()`：用 `create_progress_bar()` 替代手动 `tqdm()` 创建
- `self._pbar.write()` 替换为 `self.logger.info()` / `self.logger.warning()`
- `set_postfix_str()` 移除

#### labelme_sampler.py

- 导入：移除 `sys`、`TqdmLoggingHandler`、本地 `tqdm` 导入和 `SUPPORTED_IMAGE_EXTENSIONS`，从 `progress_logger` 导入
- `LabelMeSampler.__init__`：移除 `_tqdm_handler`，用 `setup_progress_logging()` 替代 `_setup_logging()`
- `LabelMeSampler._build_category_image_map()`：用 `create_progress_bar()` 替代手动 `tqdm()` 创建
- `self._pbar.write()` 替换为 `self.logger.warning()`
- `set_postfix_str()` 移除

#### unzip_tools.py

- 导入：移除 `sys`、`TqdmLoggingHandler`、本地 `tqdm` 导入，从 `progress_logger` 导入
- `UnzipTool.__init__`：移除 `_tqdm_handler`，用 `setup_progress_logging()` 替代 `_setup_logging()`
- `UnzipTool.extract_all()`：用 `create_progress_bar()` 替代手动 `tqdm()` 创建
- `_extract_single_file_with_pbar()` 合并入 `_extract_single_file()`，统一使用 `self.logger.warning()` 输出错误
- `self._pbar.write()` 替换为 `self.logger.warning()`
- `set_postfix_str()` 移除

#### __init__.py

- 新增导出：`TQDM_AVAILABLE`、`SUPPORTED_IMAGE_EXTENSIONS`、`setup_progress_logging`、`create_progress_bar`

## 使用方法

### 基本用法（与之前相同）

```python
from tools import clean_labelme_data

result = clean_labelme_data(
    source_dir="path/to/source",
    target_dir="path/to/target",
    log_file="path/to/log.txt",
)
```

默认 `use_tqdm=True`，控制台仅显示进度条，日志写入文件。

### 禁用进度条

```python
result = clean_labelme_data(
    source_dir="path/to/source",
    target_dir="path/to/target",
    log_file="path/to/log.txt",
    use_tqdm=False,
)
```

`use_tqdm=False` 时，控制台显示完整日志信息。

### 在 Jupyter Notebook 中使用

```python
from tools import clean_labelme_data

result = clean_labelme_data(
    source_dir="path/to/source",
    target_dir="path/to/target",
    use_tqdm=True,
)
```

自动检测 Jupyter 环境，使用 `tqdm.notebook` 渲染进度条。

### 直接使用基础设施

```python
from tools import setup_progress_logging, create_progress_bar

logger = setup_progress_logging("MyModule", "log.txt", use_tqdm=True)
pbar = create_progress_bar(total=100, desc="处理", unit="条")

for i in range(100):
    pbar.update(1)
    logger.info(f"处理第 {i+1} 条")

pbar.close()
```

## 进度条格式

统一格式：`{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]`

示例输出：`数据清洗| ██████░░░░| 50/100 [00:30<00:30, 1.67文件/s]`

更新间隔：`mininterval=1.0`，`maxinterval=5.0`，减少刷新频率避免多行输出。