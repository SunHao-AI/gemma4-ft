import logging
import os
import importlib.util
from pathlib import Path

from gemma4_core.runtime import (
    TimezoneAwareFormatter,
    configure_unsloth_compile_cache,
    format_log_timestamp,
    resolve_notebook_dir,
)


_ADAPTER_UTILS_PATH = Path(__file__).resolve().parents[1] / "distributed_training" / "adapter_utils.py"
_ADAPTER_UTILS_SPEC = importlib.util.spec_from_file_location("test_adapter_utils_module", _ADAPTER_UTILS_PATH)
_ADAPTER_UTILS = importlib.util.module_from_spec(_ADAPTER_UTILS_SPEC)
assert _ADAPTER_UTILS_SPEC.loader is not None
_ADAPTER_UTILS_SPEC.loader.exec_module(_ADAPTER_UTILS)
extract_target_modules_from_state_keys = _ADAPTER_UTILS.extract_target_modules_from_state_keys


def test_resolve_notebook_dir_prefers_explicit_env(monkeypatch, tmp_path):
    notebook_dir = tmp_path / "notebooks"
    notebook_dir.mkdir()
    monkeypatch.setenv("GEMMA4_NOTEBOOK_DIR", str(notebook_dir))

    resolved = resolve_notebook_dir(cwd=tmp_path / "other", notebook_file="")

    assert resolved == notebook_dir.resolve()


def test_resolve_notebook_dir_uses_notebook_file(tmp_path):
    notebook_dir = tmp_path / "notebooks"
    notebook_dir.mkdir()
    notebook_file = notebook_dir / "02-model_finetuning.ipynb"
    notebook_file.write_text("{}", encoding="utf-8")

    resolved = resolve_notebook_dir(cwd=tmp_path, notebook_file=str(notebook_file))

    assert resolved == notebook_dir.resolve()


def test_configure_unsloth_compile_cache_returns_absolute_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMMA4_UNSLOTH_COMPILE_CACHE_DIR", raising=False)
    cache_dir = configure_unsloth_compile_cache(tmp_path)

    assert cache_dir == (tmp_path / "unsloth_compiled_cache").resolve()
    assert cache_dir.exists()
    assert os.environ["GEMMA4_UNSLOTH_COMPILE_CACHE_DIR"] == str(cache_dir)


def test_timezone_formatter_uses_configured_timezone(monkeypatch):
    monkeypatch.setenv("GEMMA4_LOG_TIMEZONE", "UTC")
    formatter = TimezoneAwareFormatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S %Z%z")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.created = 0

    rendered = formatter.format(record)

    assert "UTC+0000" in rendered
    assert rendered.endswith(" - hello")


def test_format_log_timestamp_includes_timezone(monkeypatch):
    monkeypatch.setenv("GEMMA4_LOG_TIMEZONE", "UTC")

    timestamp = format_log_timestamp()

    assert "UTC+0000" in timestamp


def test_extract_target_modules_from_state_keys_strips_peft_prefix():
    state_keys = [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
        "base_model.model.model.layers.1.mlp.up_proj.lora_A.default.weight",
        "base_model.model.model.layers.1.mlp.up_proj.lora_B.default.weight",
        "base_model.model.model.norm.weight",
    ]

    modules = extract_target_modules_from_state_keys(state_keys)

    assert modules == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.mlp.up_proj",
    ]
