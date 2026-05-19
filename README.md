## gemma4-ft

面向 Gemma4 多模态微调与评估的应用型 ML 工程仓库。

### 仓库定位

- `src/unsloth_finetune/`：标准源码主包，作为长期维护与安装入口
- `scripts/`：训练、推理、环境检查等脚本入口
- `notebooks/`：实验、演示与分析 notebook
- `tests/`：单元测试与关键运行时校验
- `docker/`：容器化构建与运行脚本
- `docs/`：架构、迁移说明与操作文档

### 目录总览

```text
gemma4-ft/
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

### 推荐入口

- 训练：`python scripts/train_distributed.py`
- 推理：`python scripts/distributed_inference.py`
- Flash Attention 环境检查：`python scripts/check_flash_attention_env.py`
- 训练结果比较：`python scripts/compare_training_runs.py`

安装为可编辑包后，也可以直接使用：

```bash
pip install -e ".[finetune,data,monitor,dev]"
gemma4-train --help
gemma4-infer --help
```

### 架构原则

- 源码集中到 `src/unsloth_finetune/`，避免根目录平铺多包继续扩散
- Notebook 仅保留实验与展示，公共逻辑下沉到 `unsloth_finetune.notebooking`
- 训练与推理入口统一经过 `scripts/` 或 console scripts，不再直接依赖源码文件相对路径
- 根级旧包 `gemma4_core/`、`labelme_tools/`、`distributed_training/`、`color_contrast_tools/`、`notebooks/*.py` 作为兼容层保留，便于平滑迁移

### 关键文档

- `docs/project-structure-review.md`
- `docs/project-structure-guide.md`
- `ARCHITECTURE.md`

