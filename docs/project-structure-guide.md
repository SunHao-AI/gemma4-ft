# 项目结构说明指南

## 新目录布局

```text
src/unsloth_finetune/
  core/
  data/labelme/
  training/distributed/
  notebooking/
  tools/color_contrast/
scripts/
notebooks/
configs/
requirements/
tests/
docker/
docs/
```

## 各目录职责

- `src/unsloth_finetune/core/`：项目基础设施、日志、bootstrap、运行时能力
- `src/unsloth_finetune/data/labelme/`：LabelMe 数据处理、统计、采样、转换
- `src/unsloth_finetune/training/distributed/`：训练、推理、负载均衡、监控、配置
- `src/unsloth_finetune/notebooking/`：供 notebook 复用的可视化、评估和上下文初始化逻辑
- `src/unsloth_finetune/tools/color_contrast/`：弱耦合独立工具
- `scripts/`：团队推荐的本地运行入口
- `notebooks/`：实验和展示，不再沉淀核心实现
- `configs/`：版本化配置文件
- `requirements/`：补充依赖清单
- `tests/`：pytest 测试代码

## 资源放置规则

- 新增业务代码优先放入 `src/unsloth_finetune/` 对应领域目录
- Notebook 需要复用的逻辑必须下沉到 `src/unsloth_finetune/notebooking/`
- 可版本化配置统一进入 `configs/`
- 一次性运行脚本或命令入口统一进入 `scripts/`
- 训练产物、日志、缓存继续放在输出目录，不进入源码区

## 依赖约束

- `core` 为基础层，允许被所有领域依赖
- `data` 只能依赖 `core`
- `training` 可以依赖 `core` 与 `data`
- `notebooking` 可以依赖 `core`、`data`、`training`
- `tools` 尽量独立，不与主训练链路深度耦合

## 团队协作约定

- 新代码统一使用 `unsloth_finetune.*` 导入路径
- 旧路径仅视为兼容层，不作为新增代码的标准写法
- 文档中的运行命令优先指向 `scripts/` 或 console scripts
- 变更目录结构时，同时更新文档、Docker、测试和 notebook 引用

## 兼容层清理建议

以下路径后续可分阶段下线：

- `gemma4_core/`
- `labelme_tools/`
- `distributed_training/`
- `color_contrast_tools/`
- `notebooks/*.py`

建议先完成一轮发布或团队切换，再移除兼容层。

