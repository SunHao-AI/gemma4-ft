# unsloth-finetune

面向 Unsloth 框架的多模态微调与评估的应用型 ML 工程仓库（以 Gemma 4 等视觉语言模型为主要验证对象，可扩展到其他兼容模型）。

## 核心能力

- 基于 Unsloth/FastVisionModel 的多模态 LoRA/QLoRA 微调
- 分布式训练与分布式推理（torchrun + 多 rank 数据分片/动态队列）
- 训练后标准化评估、结果聚合与可选 LabelMe 导出

## 技术栈

- Python + PyTorch + Hugging Face Transformers/PEFT
- Unsloth（多模态训练与推理加速）
- Docker（可选的可复现训练/推理环境）

## 仓库定位

- `src/unsloth_finetune/`：标准源码主包，作为长期维护与安装入口
- `scripts/`：训练、推理、环境检查等脚本入口
- `notebooks/`：实验、演示与分析 notebook
- `tests/`：单元测试与关键运行时校验
- `docker/`：容器化构建与运行脚本
- `docs/`：架构、迁移说明与操作文档

## 目录总览

```text
unsloth-finetune/
├── src/unsloth_finetune/
│   ├── core/                    # 运行时基础设施
│   ├── data/labelme/            # LabelMe 数据处理与转换
│   ├── training/distributed/    # 分布式训练/推理能力
│   ├── notebooking/             # Notebook 共享辅助模块
│   └── tools/color_contrast/    # 辅助工具域
├── scripts/                     # 标准脚本入口
├── notebooks/                   # Jupyter notebooks
├── tests/                       # pytest 测试
├── configs/                     # 版本化配置资产
├── requirements/                # 附加依赖清单
├── docker/                      # 容器相关文件
└── docs/                        # 架构与说明文档
```

## 推荐入口

- 训练：`python scripts/train_distributed.py`
- 推理：`python scripts/distributed_inference.py`
- Flash Attention 环境检查：`python scripts/check_flash_attention_env.py`
- 训练结果比较：`python scripts/compare_training_runs.py`

安装为可编辑包后，也可以直接使用：

```bash
pip install -e ".[finetune,data,monitor,dev]"
unsloth-train --help
unsloth-infer --help
```

## 架构原则

- 源码集中到 `src/unsloth_finetune/`，避免根目录平铺多包继续扩散
- Notebook 仅保留实验与展示，公共逻辑下沉到 `unsloth_finetune.notebooking`
- 训练与推理入口统一经过 `scripts/` 或 console scripts，不再直接依赖源码文件相对路径
- 历史兼容层已完成迁移，当前应优先使用 `src/unsloth_finetune/`、`scripts/` 与 `.ipynb` notebook 入口

## 关键文档

- `docs/project-structure-review.md`
- `docs/project-structure-guide.md`
- `ARCHITECTURE.md`

