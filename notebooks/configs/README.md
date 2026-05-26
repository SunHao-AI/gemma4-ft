# 模型配置管理系统使用指南

## 📁 目录结构

```
notebooks/
├── configs/
│   ├── __init__.py                   # 导入接口
│   ├── config_loader.py              # 配置加载工具类
│   ├── base_config.yaml              # 基础配置模板（含完整注释）
│   ├── gemma4_e4b_config.yaml        # Gemma4 E4B 专用配置
│   ├── qwen3_5_4b_config.yaml        # Qwen3.5 4B 配置示例
│   └── llama3_2_3b_config.yaml       # Llama3.2 3B 配置示例
```

## 🚀 快速开始

### 1. 加载配置

```python
from notebooks.configs import load_model_config

# 加载 Gemma4 E4B 配置（自动验证）
config = load_model_config("gemma4_e4b")

# 访问配置参数
print(config.data_preparation.coord_norm)      # "norm_1000"
print(config.data_preparation.coord_format)    # "yxyx"
print(config.lora.r)                           # 16
print(config.training.learning_rate)           # 2e-5
```

### 2. 在 Notebook 中使用

```python
from notebooks.configs import get_config_for_notebook

# 获取完整配置字典
config_dict = get_config_for_notebook("gemma4_e4b")

# 直接使用配置参数
COORD_NORM = config_dict["data_preparation"]["coord_norm"]
COORD_FORMAT = config_dict["data_preparation"]["coord_format"]
MAX_SEQ_LENGTH = config_dict["model_loading"]["max_seq_length"]
LORA_R = config_dict["lora"]["r"]
```

### 3. 切换模型配置

只需更改配置名称即可切换不同模型：

```python
# Gemma4 配置
config = load_model_config("gemma4_e4b")

# 切换到 Qwen3.5
config = load_model_config("qwen3_5_4b")

# 切换到 Llama3.2
config = load_model_config("llama3_2_3b")
```

### 4. 动态修改配置

```python
config = load_model_config("gemma4_e4b")

# 修改单个参数
config.update({
    "training.learning_rate": 1e-5,
    "lora.r": 32,
    "distributed.num_gpus": 4,
})

# 重新验证
config.validate()
```

## 📋 配置参数说明

### 核心配置项

| 配置项 | 说明 | 可选值 |
|--------|------|--------|
| `coord_norm` | 坐标归一化模式 | `raw`, `norm_1`, `norm_100`, `norm_1000` |
| `coord_format` | 坐标输出格式 | `xyxy`, `yxyx`, `xywh`, `cxcywh` |
| `output_format` | 响应内容格式 | `labelme_text`, `box_2d_json` |
| `max_seq_length` | 最大序列长度 | 512, 1024, 2048, 4096 |
| `lora.r` | LoRA 秩 | 4-64（推荐 16-32） |
| `learning_rate` | 学习率 | 1e-6 ~ 1e-4 |

### Gemma4 特殊要求

**⚠️ 重要提醒**: Gemma4 模型有特殊的配置要求：

```yaml
data_preparation:
  coord_norm: "norm_1000"    # 必须使用 1000 归一化
  coord_format: "yxyx"       # 必须使用 yxyx 格式（box_2d 为 [y1,x1,y2,x2]）
```

这与传统格式不同：
- 传统格式: `[x_min, y_min, x_max, y_max]`
- Gemma4 格式: `[y1, x1, y2, x2]` (y坐标在前)

配置加载器会自动验证 Gemma4 模型的这些特殊要求。

## 🔧 高级用法

### 1. 查看配置摘要

```python
from notebooks.configs import print_config_summary

print_config_summary("gemma4_e4b")
```

输出示例：
```
============================================================
模型配置摘要: gemma4_e4b
============================================================

【模型信息】
  名称: gemma4_e4b
  家族: gemma
  版本: E4B

【数据准备】
  坐标归一化: norm_1000
  坐标格式: yxyx
  ...
```

### 2. 获取配置属性

```python
config = load_model_config("gemma4_e4b")

# 使用 get 方法访问（支持默认值）
value = config.get("training.learning_rate", default=2e-5)

# 支持嵌套访问
image_size = config.get("training.vision.image_width")
```

### 3. 导出配置

```python
config = load_model_config("gemma4_e4b")

# 导出为字典
config_dict = config.to_dict()

# 保存为文件
config.save("my_custom_config.yaml", format="yaml")
config.save("my_custom_config.json", format="json")
```

### 4. 列出可用配置

```python
from notebooks.configs import list_available_configs

available = list_available_configs()
print(available)  # ['base_config', 'gemma4_e4b_config', 'llama3_2_3b_config', 'qwen3_5_4b_config']
```

## 📝 配置文件格式

配置文件使用 YAML 格式，支持丰富的注释说明：

```yaml
# 数据准备配置
data_preparation:
  # 坐标归一化模式
  coord_norm: "norm_1000"
  
  # 坐标格式 - Gemma4 专用
  coord_format: "yxyx"  # box_2d 输出顺序: [y1, x1, y2, x2]
```

每个配置文件都包含详细的参数说明和取值范围注释。

## 🛡️ 配置验证

配置加载器会自动验证以下内容：

1. **参数类型验证**: 确保数值参数在合理范围内
2. **枚举值验证**: 确保枚举参数使用有效值
3. **模型特殊要求验证**: 如 Gemma4 的 coord_format 要求
4. **逻辑一致性验证**: 如不能同时启用 4-bit 和 8-bit 量化

验证失败时会抛出 `ConfigValidationError`，包含详细错误信息。

## 🔄 在数据准备 Notebook 中的集成示例

```python
# Cell 1: 导入配置
from notebooks.configs import load_model_config

# 加载 Gemma4 配置
config = load_model_config("gemma4_e4b")

# Cell 2: 使用配置参数
from unsloth_finetune.data.labelme import LabelMeConverter

converter = LabelMeConverter(
    source_dir=SOURCE_DIR,
    output_dir=OUTPUT_DIR,
    coord_norm=config.data_preparation.coord_norm,       # 自动使用正确值
    coord_format=config.data_preparation.coord_format,   # 自动使用 yxyx
    output_format=config.data_preparation.output_format,
    lang=config.data_preparation.prompt_lang,
    split=config.data_preparation.split_ratio,
    random_seed=config.data_preparation.split_seed,
)
```

## 🔄 在训练 Notebook 中的集成示例

```python
# Cell 1: 导入配置
from notebooks.configs import load_model_config

config = load_model_config("gemma4_e4b")

# Cell 2: 使用配置加载模型
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=config.model.base_model_path,
    max_seq_length=config.model_loading.max_seq_length,
    load_in_4bit=config.model_loading.load_in_4bit,
)

# Cell 3: 使用配置设置 LoRA
model = FastVisionModel.get_peft_model(
    model,
    r=config.lora.r,
    target_modules=config.lora.target_modules,
    lora_alpha=config.lora.alpha,
)
```

## 📚 相关文档

- [base_config.yaml](base_config.yaml) - 基础配置模板（含完整参数说明）
- [gemma4_e4b_config.yaml](gemma4_e4b_config.yaml) - Gemma4 E4B 专用配置
- [config_loader.py](config_loader.py) - 配置加载工具实现