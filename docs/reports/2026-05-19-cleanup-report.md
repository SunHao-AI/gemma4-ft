# 2026-05-19 项目清理报告

## 清理结论

- 本次已完成历史兼容层与 notebook 包装器的回收站迁移，尚未执行永久删除。
- 待删除资源已移动到项目根目录临时回收站：`D:\WorkPlace\Pycharm\gemma4-ft\.cleanup_trash_2026-05-19`。
- 候选资源合计：9 项，估算释放空间：87768 B（约 85.71 KiB）。
- 因当前环境验证未完全闭环，回收站保留以便复核与回滚。

## 待删除资源

- `D:\WorkPlace\Pycharm\gemma4-ft\gemma4_core` | 类型: directory | 大小: 2359 B | 最后修改: 2026-05-19 16:17:49
- `D:\WorkPlace\Pycharm\gemma4-ft\labelme_tools` | 类型: directory | 大小: 11976 B | 最后修改: 2026-05-19 15:53:59
- `D:\WorkPlace\Pycharm\gemma4-ft\distributed_training` | 类型: directory | 大小: 50233 B | 最后修改: 2026-05-19 15:53:59
- `D:\WorkPlace\Pycharm\gemma4-ft\color_contrast_tools` | 类型: directory | 大小: 21709 B | 最后修改: 2026-05-19 15:53:59
- `D:\WorkPlace\Pycharm\gemma4-ft\notebooks\common.py` | 类型: file | 大小: 287 B | 最后修改: 2026-05-19 15:53:59
- `D:\WorkPlace\Pycharm\gemma4-ft\notebooks\data_prep_shared.py` | 类型: file | 大小: 307 B | 最后修改: 2026-05-19 15:53:59
- `D:\WorkPlace\Pycharm\gemma4-ft\notebooks\eval_shared.py` | 类型: file | 大小: 297 B | 最后修改: 2026-05-19 15:53:59
- `D:\WorkPlace\Pycharm\gemma4-ft\notebooks\train_shared.py` | 类型: file | 大小: 299 B | 最后修改: 2026-05-19 15:53:59
- `D:\WorkPlace\Pycharm\gemma4-ft\notebooks\vision_shared.py` | 类型: file | 大小: 301 B | 最后修改: 2026-05-19 15:53:59

## 验证结果

- `python scripts/check_flash_attention_env.py`: 通过，生成 `flash_attention_env_report.json`。
- `python scripts/compare_training_runs.py --help`: 通过。
- `python -m pytest -q`: 通过，`405 passed in 35.65s`。
- `python scripts/train_distributed.py --help`: 未通过，当前 Windows + Python 3.13 环境中异常退出。
- `python scripts/distributed_inference.py --help`: 未通过，当前 Windows + Python 3.13 环境中异常退出。
- `docker build -f docker/Dockerfile -t unsloth-finetune:cleanup-check .`: 未执行成功，Docker Desktop Linux Engine 未启动。

## 已完成的清理前置修复

- 将测试、notebook 与 Docker 文档中的旧兼容层路径改为 `unsloth_finetune.*`、`scripts/` 与新 requirements 路径。
- 为 `scripts/*.py` 增加项目根路径引导，支持从仓库根直接执行脚本。
- 保留 `REFACTOR.md` 作为历史重构记录，不纳入删除范围。

## 阻塞项

- 需要在可用的 Docker daemon 环境下补跑镜像构建。
- 需要确认训练/推理脚本在目标运行环境中的异常退出是否为既有环境问题；在未确认前不建议永久删除回收站。

## 永久删除建议

- 先在目标训练环境补跑 Docker 构建与 `train_distributed.py` / `distributed_inference.py` 帮助命令。
- 若补充验证通过，再删除 `D:\WorkPlace\Pycharm\gemma4-ft\.cleanup_trash_2026-05-19` 完成最终清理。
