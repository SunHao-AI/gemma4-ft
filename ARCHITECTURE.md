# gemma4-ft 项目架构文档

## 项目定位

`gemma4-ft` 采用应用型 ML 工程仓库结构，服务于以下场景：

- LabelMe 数据清洗、统计、采样与格式转换
- Gemma4 / Unsloth 多模态训练与分布式推理
- Notebook 演示、实验编排与效果分析
- Docker 化本地开发和训练环境

本次结构重构后，源码统一收口到 `src/gemma4_ft/`，根级历史包保留为兼容层，仅用于过渡。

## 目录结构

```text
gemma4-ft/
├── src/gemma4_ft/
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
├── docs/                            # 架构与迁移文档
├── gemma4_core/                     # 兼容层
├── labelme_tools/                   # 兼容层
├── distributed_training/            # 兼容层
└── color_contrast_tools/            # 兼容层
```

## 分层职责

### `gemma4_ft.core`

- 管理 notebook bootstrap、项目根定位、日志与时间格式
- 管理 Unsloth 编译缓存与 LabelMe 推理结果导出
- 作为跨训练、数据、notebook 场景复用的基础设施层

### `gemma4_ft.data.labelme`

- 负责 LabelMe 数据清洗、统计、采样、转换和压缩包解压
- 不直接依赖训练编排逻辑
- 作为数据准备层，可被 notebook 与训练层共同使用

### `gemma4_ft.training.distributed`

- 管理训练配置、多 GPU 训练、推理、负载均衡与监控
- 负责训练/推理主流程编排
- 依赖 `core` 和部分 `data` 基础能力，但不依赖 notebook 层

### `gemma4_ft.notebooking`

- 存放 notebook 共享帮助函数、可视化与评估辅助逻辑
- 只承载实验展示层复用代码
- 不应成为核心业务逻辑的唯一实现位置

### `gemma4_ft.tools.color_contrast`

- 存放与主训练链路弱耦合的独立工具
- 便于后续单独维护或迁出

## 依赖关系

```text
gemma4_ft.core
    ├─→ gemma4_ft.data.labelme
    ├─→ gemma4_ft.training.distributed
    └─→ gemma4_ft.notebooking

gemma4_ft.data.labelme
    └─→ gemma4_ft.training.distributed

gemma4_ft.training.distributed
    └─→ gemma4_ft.notebooking  (禁止反向依赖)

gemma4_ft.tools.color_contrast
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

- `gemma4-train`
- `gemma4-infer`
- `gemma4-check-flash-attention`
- `gemma4-compare-runs`

## 兼容策略

以下目录在重构后保留为兼容层：

- `gemma4_core/`
- `labelme_tools/`
- `distributed_training/`
- `color_contrast_tools/`
- `notebooks/*.py`

兼容层仅做转发导入，后续应逐步清理业务代码中的旧路径引用。
