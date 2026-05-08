# 模型对比测试系统使用指南

## 概述

本系统用于系统性地对比微调后的 Gemma 4 模型与原始权重模型在目标检测任务上的性能。测试包括两个维度:

1. **数值量化对比**: 计算 IOU(交并比)指标
2. **可视化效果对比**: 生成检测结果对比图像

## 系统要求

- Python 3.10+
- CUDA 11.8+ 或 12.1+
- GPU VRAM ≥ 10GB (E4B 模型)
- 必要依赖包:
  - torch
  - pillow
  - matplotlib
  - requests
  - numpy

## 快速开始

### 1. 安装依赖

```bash
pip install torch pillow matplotlib requests numpy
```

### 2. 配置测试参数

编辑 `comparison_config.json` 文件,配置以下内容:

#### 模型配置

```json
{
  "models": {
    "base_model": {
      "name": "原始模型",
      "base_model_path": "/path/to/base/model",
      "max_seq_length": 2048,
      "load_in_4bit": true,
      "device_map": "cuda:0"
    },
    "finetuned_model": {
      "name": "微调模型",
      "base_model_path": "/path/to/base/model",
      "lora_adapter_path": "/path/to/lora/adapter",
      "max_seq_length": 2048,
      "load_in_4bit": true,
      "device_map": "cuda:0"
    }
  }
}
```

#### 测试数据配置

```json
{
  "test_data": {
    "images": [
      {
        "source": "https://example.com/image.jpg",
        "query": "检测图中的目标类别",
        "ground_truth": [
          {
            "bbox": [x1, y1, x2, y2],
            "label": "目标类别",
            "confidence": 1.0
          }
        ]
      }
    ],
    "confidence_thresholds": [0.5, 0.7, 0.85, 0.95]
  }
}
```

### 3. 准备测试数据集

测试数据集需要包含:

- **图像来源**: 支持本地路径和网络 URL
- **检测查询**: 明确的目标检测任务描述
- **真实标注(Ground Truth)**: 用于计算 IOU 的标准答案

#### Ground Truth 格式说明

```json
{
  "bbox": [x1, y1, x2, y2],  // 边界框坐标(像素坐标)
  "label": "目标类别",       // 目标类别标签
  "confidence": 1.0          // 真实标注置信度通常为 1.0
}
```

### 4. 运行测试

```bash
python model_comparison_test.py --config comparison_config.json --output_dir ./comparison_results
```

## 测试指标说明

### IOU(交并比)

IOU 用于评估检测框的准确性,计算公式:

```
IOU = 交集面积 / 并集面积
```

- IOU = 1.0: 完全匹配
- IOU = 0.0: 完全不匹配
- IOU ∈ [0, 1]: 部分匹配

### 测试输出指标

系统会计算以下指标:

#### 1. 整体 IOU 统计

- **平均 IOU**: 所有检测的平均值
- **标准差**: IOU 分布的离散程度
- **最小/最大 IOU**: IOU 范围

#### 2. 类别级 IOU 统计

对每个目标类别单独统计 IOU,便于分析模型在不同类别上的表现。

#### 3. 置信度阈值分析

在不同置信度阈值下统计:
- 检测数量
- IOU 表现

帮助分析模型在不同置信度要求下的表现。

#### 4. 性能改进指标

- **IOU 提升**: 微调模型相对于原始模型的绝对提升
- **改进百分比**: 相对原始模型的改进率

## 输出文件

### 1. 测试报告 (JSON)

文件名: `comparison_report_YYYYMMDD_HHMMSS.json`

包含:
- 测试概况
- 整体 IOU 统计
- 类别级分析
- 置信度阈值分析
- 详细测试结果

### 2. 对比可视化图像

文件名: `comparison_<image_name>.png`

包含:
- 原始模型检测结果
- 微调模型检测结果
- IOU 统计信息

## 自定义测试数据集

### 使用本地图像

```json
{
  "source": "/path/to/local/image.jpg",
  "query": "检测图中的人",
  "ground_truth": [
    {
      "bbox": [100, 200, 300, 400],
      "label": "person",
      "confidence": 1.0
    }
  ]
}
```

### 使用网络图像

```json
{
  "source": "https://example.com/image.jpg",
  "query": "检测图中的汽车",
  "ground_truth": [
    {
      "bbox": [50, 100, 250, 200],
      "label": "car",
      "confidence": 1.0
    }
  ]
}
```

### 多目标检测

```json
{
  "source": "/path/to/image.jpg",
  "query": "检测图中的人、车、狗",
  "ground_truth": [
    {
      "bbox": [100, 200, 300, 400],
      "label": "person",
      "confidence": 1.0
    },
    {
      "bbox": [400, 150, 600, 350],
      "label": "car",
      "confidence": 1.0
    },
    {
      "bbox": [700, 500, 850, 650],
      "label": "dog",
      "confidence": 1.0
    }
  ]
}
```

## 测试结果解读

### 好的测试结果

- 平均 IOU > 0.7: 检测框定位准确
- 微调模型 IOU 显著高于原始模型
- 类别级 IOU 分布均匀

### 需要改进的结果

- 平均 IOU < 0.5: 检测框定位不够准确
- 微调模型改进不明显
- 特定类别 IOU 较低

## 高级功能

### 1. 批量测试

在配置文件中添加多个测试样本:

```json
{
  "test_data": {
    "images": [
      {"source": "image1.jpg", ...},
      {"source": "image2.jpg", ...},
      {"source": "image3.jpg", ...}
    ]
  }
}
```

### 2. 自定义置信度阈值

调整置信度阈值以适应不同场景:

```json
{
  "confidence_thresholds": [0.3, 0.5, 0.7, 0.9]
}
```

### 3. 多类别测试

测试模型在多个类别上的表现:

```json
{
  "query": "检测图中的所有目标,包括人、车、建筑"
}
```

## 常见问题

### Q: 模型加载失败?

检查:
- 模型路径是否正确
- GPU 内存是否足够
- 是否安装了必要的依赖

### Q: IOU 为 0?

可能原因:
- 模型未检测到目标
- Ground Truth 标注不准确
- 坐标格式错误

### Q: 如何提高测试准确性?

建议:
- 使用多样化的测试数据集
- 确保 Ground Truth 标注准确
- 测试不同置信度阈值

## 测试建议

### 1. 数据集多样性

- 包含不同场景
- 包含不同目标类别
- 包含不同光照条件

### 2. Ground Truth 准确性

- 使用专业标注工具
- 多人交叉验证
- 标注边界清晰

### 3. 测试指标全面性

- 测试多个置信度阈值
- 分析类别级表现
- 对比可视化结果

## 示例运行

```bash
# 使用默认配置
python model_comparison_test.py

# 指定配置文件和输出目录
python model_comparison_test.py --config my_config.json --output_dir ./my_results

# 在 Windows PowerShell 中运行
python model_comparison_test.py --config comparison_config.json --output_dir ./comparison_results
```

## 输出示例

### 测试报告摘要

```
测试概况:
  总测试数: 10
  成功测试: 9
  失败测试: 1

整体IOU表现 (相对于真实标注):
  原始模型:
    平均IOU: 0.65 ± 0.12
    IOU范围: [0.45, 0.88]
  微调模型:
    平均IOU: 0.78 ± 0.08
    IOU范围: [0.62, 0.92]

性能改进:
  IOU提升: 0.13
  改进百分比: 20.0%
```

### 类别级分析

```
各类别IOU表现:
  person:
    原始模型: 0.68
    微调模型: 0.82
    改进: 0.14
  car:
    原始模型: 0.62
    微调模型: 0.74
    改进: 0.12
```

### 置信度阈值分析

```
不同置信度阈值下的表现:
  置信度阈值 0.5:
    原始模型: IOU=0.60, 平均检测数=5.2
    微调模型: IOU=0.72, 平均检测数=4.8
  置信度阈值 0.85:
    原始模型: IOU=0.70, 平均检测数=3.1
    微调模型: IOU=0.85, 平均检测数=2.8
```

## 技术支持

如有问题,请检查:
1. 配置文件格式是否正确
2. 模型路径是否存在
3. 测试数据是否可访问
4. GPU 资源是否充足

## 许可证

本测试系统仅供学习和研究使用。Gemma 模型使用受 Google 许可证约束。