import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "02-model_finetuning.ipynb"


def _load_notebook_source() -> str:
    data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    lines = []
    for cell in data.get("cells", []):
        lines.extend(cell.get("source", []))
    return "".join(lines)


def test_multi_gpu_defaults_are_conservative():
    source = _load_notebook_source()

    assert 'MULTI_GPU_BATCH_SIZE = 4' in source
    assert 'MULTI_GPU_LR_BASE = 2e-5' in source
    assert 'MULTI_GPU_LR_SCALING = "none"' in source
    assert 'MULTI_GPU_WARMUP_RATIO = 0.1' in source


def test_latest_marker_written_only_after_success():
    source = _load_notebook_source()

    assert '_train_exit_code = get_ipython().system(ddp_cmd)' in source
    assert 'if _train_exit_code == 0:' in source
    assert 'print(f"训练命令退出码: {_train_exit_code}，跳过写入latest.txt")' in source


def test_training_preflight_cell_exists():
    source = _load_notebook_source()

    assert "训练前检查" in source
    assert "effective_lr过高" in source
