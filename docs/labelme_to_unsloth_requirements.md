# LabelMe -> Unsloth 多模态目标检测数据转换需求文档

**版本**: v1.0
**日期**: 2025-05-22
**状态**: 已实现
**实现模块**:
- `src/unsloth_finetune/data/labelme/labelme_converter.py`
- `src/unsloth_finetune/data/labelme/detection_format.py`
- `src/unsloth_finetune/tools/labelme_to_training_format.py`

---

## 目录

1. [项目背景](#1-项目背景)
2. [坐标归一化策略](#2-坐标归一化策略)
3. [坐标格式选项](#3-坐标格式选项)
4. [输出数据格式](#4-输出数据格式)
5. [数据生成策略](#5-数据生成策略)
6. [数据集划分](#6-数据集划分)
7. [Prompt 模板配置](#7-prompt-模板配置)
8. [数据过滤与校验](#8-数据过滤与校验)
9. [图片处理选项](#9-图片处理选项)
10. [统计信息输出](#10-统计信息输出)
11. [待确认事项](#11-待确认事项)

---

## 1. 项目背景

### 目标

将 **LabelMe 格式**的目标检测标注数据, 转换为适配 **Unsloth 多模态微调**的对话式训练数据.

### 数据流向

```
LabelMe JSON 标注文件
        |
        v
  转换脚本 (本项目)
        |
        v
ShareGPT / Alpaca 格式 JSONL
  |- train.jsonl
  |- val.jsonl
  └- test.jsonl  (仅 human 侧, 无 answer)
```

### LabelMe 输入格式

```json
{
  "version": "5.0.1",
  "imagePath": "img_001.jpg",
  "imageUrl": "",
  "imageHeight": 480,
  "imageWidth": 640,
  "shapes": [
    {
      "label": "cat",
      "shape_type": "rectangle",
      "points": [[120.0, 45.0], [320.0, 280.0]]
    },
    {
      "label": "dog",
      "shape_type": "rectangle",
      "points": [[400.0, 100.0], [650.0, 390.0]]
    }
  ]
}
```

---

## 2. 坐标归一化策略

支持以下归一化模式, 通过参数 `--coord-norm` 指定:

| 模式 | 参数值 | 说明 | 示例 (原始: x=120, img_w=640) | 适用模型 |
|------|--------|------|-----------------------------|----------|
| 原始像素坐标 | `raw` | 不做任何处理, 保留整数像素值 | `120` | 通用 baseline |
| 归一化到 [0, 1] | `norm_1` | 除以图片宽/高, 保留 4 位小数 | `0.1875` | LLaVA 等 |
| 归一化到 [0, 100] | `norm_100` | 百分比坐标, 取整 | `19` | 自定义模型 |
| 归一化到 [0, 1000] | `norm_1000` | 乘以 1000 后取整 | `188` | Qwen-VL, InternVL |

**计算公式:**

```
norm_1    : x_norm = round(x / img_width, 4)
norm_100  : x_norm = round(x / img_width * 100)
norm_1000 : x_norm = round(x / img_width * 1000)
```

---

## 3. 坐标格式选项

支持以下坐标表示格式, 通过参数 `--coord-format` 指定:

| 格式 | 参数值 | 说明 | 示例 |
|------|--------|------|------|
| 左上 + 右下 | `xyxy` | `[x1, y1, x2, y2]` | `[120, 45, 320, 280]` |
| 左上 + 宽高 | `xywh` | `[x, y, width, height]` | `[120, 45, 200, 235]` |
| 中心点 + 宽高 | `cxcywh` | `[cx, cy, width, height]` | `[220, 162, 200, 235]` |

> **注意**: 归一化策略与坐标格式可任意组合, 例如 `norm_1000 + xyxy` 或 `norm_1 + cxcywh`.

---

## 4. 输出数据格式

输出为 **ShareGPT 格式** JSONL 文件 (每行一条 JSON 记录).

### 4.1 训练/验证格式 (含 answer)

#### 每图全类别合并 (all-in-one, 推荐)

```json
{
  "id": "img_001",
  "image": "images/img_001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nPlease detect all [cat, dog] in the image and return their bounding boxes."
    },
    {
      "from": "gpt",
      "value": "I detected the following objects:\n- cat: [120, 45, 320, 280]\n- dog: [400, 100, 650, 390]"
    }
  ]
}
```

#### 每图每类一条 (per-class)

```json
{
  "id": "img_001_cat",
  "image": "images/img_001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nDetect all [cat] in the image and return their bounding boxes."
    },
    {
      "from": "gpt",
      "value": "I found 1 [cat]:\n- [120, 45, 320, 280]"
    }
  ]
}

{
  "id": "img_001_dog",
  "image": "images/img_001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nDetect all [dog] in the image and return their bounding boxes."
    },
    {
      "from": "gpt",
      "value": "I found 1 [dog]:\n- [400, 100, 650, 390]"
    }
  ]
}
```

#### 多实例同类示例

```json
{
  "id": "img_002_person",
  "image": "images/img_002.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nDetect all [person] in the image and return their bounding boxes."
    },
    {
      "from": "gpt",
      "value": "I found 3 [person]:\n- [50, 30, 200, 400]\n- [210, 45, 380, 410]\n- [420, 60, 590, 420]"
    }
  ]
}
```

### 4.2 测试/推理格式 (仅 human 侧, 无 answer)

```json
{
  "id": "img_001",
  "image": "images/img_001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nPlease detect all [{class}, {class}, ..., {class}] in the image and return their bounding boxes."
    }
  ]
}
```

### 4.3 输出文件结构

```
output_dir/
  |- images/              # 图片文件 (可选复制)
  |- train.jsonl
  |- valid.jsonl
  |- test.jsonl
  └- dataset_info.json    # 统计信息
```

---

## 5. 数据生成策略

通过参数 `--gen-strategy` 指定:

| 策略 | 参数值 | 说明 | 优点 | 缺点 |
|------|--------|------|------|------|
| 每图一条 | `all_in_one` | 每张图片的所有目标合并到一条记录 | 数据量精简, 任务完整, 接近真实推理 | 长序列, 多目标时训练难度稍高 |
| 每图每类一条 | `per_class` | 每张图片按类别拆分, 每类一条记录 | 数据量扩增, 类别针对性强 | 同图重复加载, 语义碎片化 |
| 两种都生成 | `both` | 同时输出以上两种格式到不同文件 | 灵活组合训练 | 需自行管理文件, 避免数据泄露 |

---

## 6. 数据集划分

### 6.1 划分比例

通过参数 `--split` 指定, 格式为 `train:val:test`, 默认 `8:1:1`.

```bash
--split 8:1:1   # 80% 训练, 10% 验证, 10% 测试 (默认)
--split 9:1:0   # 90% 训练, 10% 验证, 不生成测试集
--split 7:2:1
```

### 6.2 划分方式

通过参数 `--split-method` 指定:

| 方式 | 参数值 | 说明 |
|------|--------|------|
| 随机划分 | `random` | 随机打乱后按比例切分, 可配合 `--seed` 固定随机种子 |
| 顺序划分 | `sequential` | 按文件名排序后顺序切分 |
| 分层采样 | `stratified` | 按类别分层, 保证各 split 的类别分布均匀 (推荐多类别场景) |

---

## 7. Prompt 模板配置

### 7.1 语言选项

通过参数 `--lang` 指定:

| 参数值 | human 侧 prompt 示例 |
|--------|---------------------|
| `en` (默认) | `Please detect all [{class}, {class}, ..., {class}] in the image and return their bounding boxes.` |
| `zh` | `请检测图片中所有的[{class}, {class}, ..., {class}], 并返回它们的边界框坐标.` |

### 7.2 Prompt 风格

通过参数 `--prompt-style` 指定:

| 风格 | 参数值 | human 侧 prompt 示例 |
|------|--------|---------------------|
| 简洁型 | `simple` | `Detect all [{class}, {class}, ..., {class}].` |
| 描述型 | `descriptive` | `Please detect all [{class}, {class}, ..., {class}] in the image and return their categories and bounding boxes.` |
| CoT 型 | `cot` | `Please think step by step, then detect all [{class}, {class}, ..., {class}] in the image and return their bounding boxes.` |

### 7.3 自定义模板

支持通过 `--prompt-template` 传入自定义模板文件 (YAML 格式), 使用 `{class}` 占位符:

```yaml
# custom_prompt.yaml
all_in_one_en: "Please detect all [{class}, {class}, ..., {class}] in the image and return their bounding boxes."
per_class_en: "Detect all [{class}] in the image and return their bounding boxes."
all_in_one_zh: "请检测图片中所有[{class}, {class}, ..., {class}], 并返回边界框坐标."
per_class_zh: "请检测图片中所有的 [{class}], 并返回边界框坐标."
```

---

## 8. 数据过滤与校验

### 8.1 空标注过滤

- 自动跳过没有任何 `shapes` 的图片 (空标注文件)
- 可通过 `--keep-empty` 参数保留空标注图片 (用于负样本训练)
- 提供polygon转rectangle选项

### 8.2 bbox 异常校验

自动过滤以下异常标注框:

| 异常类型 | 判断条件 |
|----------|----------|
| 坐标负值 | `x1 < 0` 或 `y1 < 0` |
| 超出图片边界 | `x2 > img_width` 或 `y2 > img_height` |
| 宽高为零 | `x2 <= x1` 或 `y2 <= y1` |
| 面积过小 | 宽或高小于阈值 (默认 `--min-bbox-size 2`) |

### 8.3 类别过滤

```bash
# 白名单: 只保留指定类别
--class-whitelist cat,dog,person

# 黑名单: 排除指定类别
--class-blacklist background,ignore

# 不过滤 (默认)
```

### 8.4 类别重映射

通过 `--class-remap` 传入映射文件 (JSON), 将多个标签统一为标准名称:

```json
{
  "human": "person",
  "people": "person",
  "automobile": "car",
  "vehicle": "car"
}
```

### 8.5 标注类型过滤

```bash
# 只处理矩形框 (默认)
--shape-types rectangle

# 同时处理多边形 (自动转换为外接矩形 bbox)
--shape-types rectangle,polygon
```

---

## 9. 图片处理选项

| 选项 | 参数 | 说明 |
|------|------|------|
| 复制图片到输出目录 | `--copy-images` | 将原始图片复制到 `output_dir/images/` |
| 仅保留相对路径 | `--image-path relative` | 输出相对于 JSONL 文件的路径 (默认) |
| 保留绝对路径 | `--image-path absolute` | 输出图片的绝对路径 |
| 内嵌 base64 | `--image-path base64` | 将图片编码为 base64 内嵌到 JSON (小数据集, 不推荐大规模使用) |

---

## 10. 统计信息输出

转换完成后自动生成 `dataset_info.json`, 包含以下内容:

```json
{
  "total_images": 1000,
  "total_annotations": 4523,
  "splits": {
    "train": { "images": 800, "annotations": 3620, "records": 800 },
    "valid":   { "images": 100, "annotations": 452,  "records": 100 },
    "test":  { "images": 100, "annotations": 451,  "records": 100 }
  },
  "class_distribution": {
    "cat":    { "total": 1200, "train": 960, "val": 120, "test": 120 },
    "dog":    { "total": 980,  "train": 784, "val": 98,  "test": 98  },
    "person": { "total": 2343, "train": 1876, "val": 234, "test": 233 }
  },
  "config": {
    "coord_norm": "norm_1000",
    "coord_format": "xyxy",
    "gen_strategy": "all_in_one",
    "split": "8:1:1",
    "split_method": "stratified",
    "lang": "en",
    "prompt_style": "descriptive"
  },
  "skipped": {
    "empty_annotations": 5,
    "invalid_bbox": 12,
    "missing_images": 2
  }
}
```

---

## 11. 待确认事项

在正式实现前, 请确认以下问题:

| # | 问题 | 选项 | 你的选择 |
|---|------|------|---------|
| 1 | **目标模型** | Qwen2-VL / LLaVA / InternVL / Phi-3.5-Vision / 其他 | Gemma4 |
| 2 | **LabelMe 标注类型** | 只有矩形框 / 包含多边形 (需转 bbox) | 包含多边形 (需转 bbox) |
| 3 | **输出文件格式** | `.jsonl` (每行一条, 推荐) / `.json` (整个列表) | .jsonl |
| 4 | **Prompt 语言** | 中文 / 英文 / 两者都要 | 两者都要 |
| 5 | **图片路径方式** | 相对路径 / 绝对路径 / base64 内嵌 | 相对路径 / 绝对路径 / base64 内嵌 |
| 6 | **默认坐标归一化** | raw / norm_1 / norm_100 / norm_1000 | norm_1000 |
| 7 | **默认生成策略** | all_in_one / per_class / both | all_in_one |

> 确认以上事项后, 将基于本文档生成完整的转换脚本.
