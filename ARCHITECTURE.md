# unsloth-finetune 项目架构文档

## 项目定位

`unsloth-finetune` 采用应用型 ML 工程仓库结构，服务于以下场景：

- LabelMe 数据清洗、统计、采样与格式转换
- 基于 Unsloth 的多模态微调与分布式推理（以 Gemma 4 等模型为示例）
- Notebook 演示、实验编排与效果分析
- Docker 化本地开发和训练环境

本次结构重构后，源码统一收口到 `src/unsloth_finetune/`，根级历史包保留为兼容层，仅用于过渡。

## 目录结构

```text
unsloth-finetune/
├── src/unsloth_finetune/
│   ├── core/                         # 运行时基础设施与项目自举
│   ├── data/
│   │   └── labelme/                 # LabelMe 数据处理领域
│   ├── training/
│   │   └── distributed/             # 分布式训练/推理领域
│   ├── notebooking/                 # Notebook 共享模块
│   └── tools/
│       └── color_contrast/          # 独立辅助工具域
├── scripts/                         # 标准脚本入口
├── notebooks/                       # 仅存放 .ipynb 与兼容包装模块
├── tests/                           # pytest 测试
├── configs/                         # 版本化配置资源
├── requirements/                    # 附加依赖清单
├── docker/                          # 容器构建与运行脚本
└── docs/                            # 架构与迁移文档
```

## 分层职责

### `unsloth_finetune.core`

- 管理 notebook bootstrap、项目根定位、日志与时间格式
- 管理 Unsloth 编译缓存与 LabelMe 推理结果导出
- 作为跨训练、数据、notebook 场景复用的基础设施层

### `unsloth_finetune.data.labelme`

- 负责 LabelMe 数据清洗、统计、采样、转换和压缩包解压
- 不直接依赖训练编排逻辑
- 作为数据准备层，可被 notebook 与训练层共同使用

### `unsloth_finetune.training.distributed`

- 管理训练配置、多 GPU 训练、推理、负载均衡与监控
- 负责训练/推理主流程编排
- 依赖 `core` 和部分 `data` 基础能力，但不依赖 notebook 层

### `unsloth_finetune.notebooking`

- 存放 notebook 共享帮助函数、可视化与评估辅助逻辑
- 只承载实验展示层复用代码
- 不应成为核心业务逻辑的唯一实现位置

### `unsloth_finetune.tools.color_contrast`

- 存放与主训练链路弱耦合的独立工具
- 便于后续单独维护或迁出

## 依赖关系

```text
unsloth_finetune.core
    ├─→ unsloth_finetune.data.labelme
    ├─→ unsloth_finetune.training.distributed
    └─→ unsloth_finetune.notebooking

unsloth_finetune.data.labelme
    └─→ unsloth_finetune.training.distributed

unsloth_finetune.training.distributed
    └─→ unsloth_finetune.notebooking  (禁止反向依赖)

unsloth_finetune.tools.color_contrast
    └─→ 独立维护，不参与主链路
```

约束如下：

- `training` 可以使用 `core` 与 `data`，但不应依赖 `notebooking`
- `notebooking` 可以调用 `core`、`data`、`training` 的公开接口
- `notebooks/` 中的 `.ipynb` 只做编排与展示，不沉淀核心实现

## 运行入口

推荐使用以下入口，而不是直接运行 `src/` 下源码文件：

- `python scripts/train_distributed.py`
- `python scripts/distributed_inference.py`
- `python scripts/check_flash_attention_env.py`
- `python scripts/compare_training_runs.py`

安装项目后，也可使用 console scripts：

- `unsloth-train`
- `unsloth-infer`
- `unsloth-check-flash-attention`
- `unsloth-compare-runs`

## 清理状态

以下旧兼容层已完成迁移并已删除：

- `gemma4_core/`（已删除）
- `labelme_tools/`（已删除）
- `distributed_training/`（已删除）
- `color_contrast_tools/`（已删除）
- `gemma4_multimodal_demo/`（已删除）

根目录 `unsloth_finetune/` shim 当前保留作为过渡兼容层，建议在稳定运行后删除。

