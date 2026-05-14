"""
测试 progress_logger 模块的核心功能
覆盖 setup_progress_logging, create_progress_bar, PhaseProgressManager,
print_phase_header, print_phase_footer, SUPPORTED_IMAGE_EXTENSIONS 等
"""

import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from labelme_tools.progress_logger import (
    IN_NOTEBOOK,
    SUPPORTED_IMAGE_EXTENSIONS,
    TQDM_AVAILABLE,
    PhaseProgressManager,
    create_progress_bar,
    print_phase_footer,
    print_phase_header,
    setup_progress_logging,
)


class TestSetupProgressLogging:

    def test_returns_logger_instance(self):
        logger = setup_progress_logging("test_basic", use_tqdm=False)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_basic"

    def test_logger_propagate_false(self):
        logger = setup_progress_logging("test_propagate", use_tqdm=False)
        assert logger.propagate is False

    def test_console_handler_when_no_tqdm(self):
        logger = setup_progress_logging("test_console", use_tqdm=False)
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_no_console_handler_when_tqdm(self):
        logger = setup_progress_logging("test_tqdm_mode", use_tqdm=True)
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 0

    def test_null_handler_when_tqdm_no_log_file(self):
        logger = setup_progress_logging("test_null", use_tqdm=True)
        null_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.NullHandler)
        ]
        assert len(null_handlers) == 1

    def test_file_handler_with_log_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        logger = setup_progress_logging(
            "test_file", log_file=str(log_file), use_tqdm=False
        )
        file_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert log_file.exists()

    def test_log_file_auto_created_dir(self, tmp_path):
        log_file = tmp_path / "subdir" / "deep" / "test.log"
        logger = setup_progress_logging(
            "test_deep_dir", log_file=str(log_file), use_tqdm=False
        )
        assert log_file.parent.exists()

    def test_logger_clears_old_handlers(self):
        logger1 = setup_progress_logging("test_clear", use_tqdm=False)
        initial_count = len(logger1.handlers)
        logger2 = setup_progress_logging("test_clear", use_tqdm=False)
        assert len(logger2.handlers) == initial_count

    def test_log_level_setting(self):
        logger = setup_progress_logging(
            "test_level", use_tqdm=False, log_level=logging.DEBUG
        )
        assert logger.level == logging.DEBUG

    def test_formatter_format(self, tmp_path):
        log_file = tmp_path / "fmt.log"
        logger = setup_progress_logging(
            "test_fmt", log_file=str(log_file), use_tqdm=False
        )
        logger.info("test message")
        for handler in logger.handlers:
            handler.close()
        content = log_file.read_text(encoding="utf-8")
        assert "test message" in content


class TestCreateProgressBar:

    def test_returns_none_when_tqdm_not_available(self):
        with mock.patch("labelme_tools.progress_logger.TQDM_AVAILABLE", False):
            result = create_progress_bar(total=10, desc="test")
            assert result is None

    def test_returns_progress_bar_when_tqdm_available(self):
        if not TQDM_AVAILABLE:
            pytest.skip("tqdm 未安装")
        pbar = create_progress_bar(total=10, desc="test", unit="项")
        assert pbar is not None
        pbar.close()

    def test_custom_parameters(self):
        if not TQDM_AVAILABLE:
            pytest.skip("tqdm 未安装")
        pbar = create_progress_bar(
            total=100,
            desc="custom",
            unit="条",
            mininterval=0.1,
            maxinterval=10.0,
        )
        assert pbar is not None
        pbar.close()


class TestSUPPORTED_IMAGE_EXTENSIONS:

    def test_contains_common_extensions(self):
        assert ".jpg" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".jpeg" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".png" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".bmp" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".gif" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".tif" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".tiff" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".webp" in SUPPORTED_IMAGE_EXTENSIONS

    def test_is_set_type(self):
        assert isinstance(SUPPORTED_IMAGE_EXTENSIONS, set)

    def test_all_extensions_start_with_dot(self):
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            assert ext.startswith(".")


class TestPhaseProgressManager:

    def test_init_with_phases(self):
        phases = ["validation", "deduplication", "copy"]
        manager = PhaseProgressManager(phases, use_tqdm=False)
        assert manager.phases == phases
        assert manager.current_phase_index == -1
        assert len(manager.phase_info) == 3

    def test_phase_info_initialized(self):
        phases = ["validation", "copy"]
        manager = PhaseProgressManager(phases, use_tqdm=False)
        assert manager.phase_info["validation"]["total"] == 0
        assert manager.phase_info["validation"]["completed"] == 0
        assert manager.phase_info["validation"]["display_name"] == "数据验证"

    def test_display_name_mapping(self):
        phases = ["validation"]
        manager = PhaseProgressManager(phases, use_tqdm=False)
        assert manager._get_display_name("validation") == "数据验证"
        assert manager._get_display_name("unknown_phase") == "unknown_phase"

    def test_start_phase(self, capsys):
        phases = ["validation", "copy"]
        manager = PhaseProgressManager(phases, use_tqdm=False)
        manager.start_phase("validation", 10)
        assert manager.current_phase_index == 0
        assert manager.phase_info["validation"]["total"] == 10
        assert manager.phase_info["validation"]["start_time"] is not None
        captured = capsys.readouterr()
        assert "数据验证" in captured.out

    def test_update_phase(self, capsys):
        phases = ["validation"]
        manager = PhaseProgressManager(phases, use_tqdm=False)
        manager.start_phase("validation", 10)
        manager.update(3)
        assert manager.phase_info["validation"]["completed"] == 3
        captured = capsys.readouterr()
        assert "3/10" in captured.out

    def test_update_before_start(self):
        phases = ["validation"]
        manager = PhaseProgressManager(phases, use_tqdm=False)
        manager.update(1)
        # Should not crash; no current phase
        assert manager.current_phase_index == -1

    def test_complete_phase(self, capsys):
        phases = ["validation", "copy"]
        manager = PhaseProgressManager(phases, use_tqdm=False)
        manager.start_phase("validation", 5)
        manager.update(5)
        manager.complete_phase("validation")
        assert manager.phase_info["validation"]["end_time"] is not None
        captured = capsys.readouterr()
        assert "完成" in captured.out

    def test_complete_current_phase(self, capsys):
        phases = ["validation"]
        manager = PhaseProgressManager(phases, use_tqdm=False)
        manager.start_phase("validation", 3)
        manager.update(3)
        manager.complete_phase()
        assert manager.current_phase_index == -1

    def test_complete_all(self, capsys):
        phases = ["validation"]
        manager = PhaseProgressManager(phases, use_tqdm=False)
        manager.start_phase("validation", 5)
        manager.update(5)
        manager.complete_phase("validation")
        summary = manager.complete_all(show_summary=True)
        assert "total_duration" in summary
        assert "phases_completed" in summary
        assert "phase_details" in summary
        captured = capsys.readouterr()
        assert "全部任务完成" in captured.out

    def test_complete_all_no_summary(self, capsys):
        phases = ["validation"]
        manager = PhaseProgressManager(phases, use_tqdm=False)
        manager.start_phase("validation", 5)
        manager.update(5)
        manager.complete_phase("validation")
        summary = manager.complete_all(show_summary=False)
        assert "total_duration" in summary
        captured = capsys.readouterr()
        assert "全部任务完成" not in captured.out

    def test_get_progress_info_no_phase(self):
        phases = ["validation"]
        manager = PhaseProgressManager(phases, use_tqdm=False)
        info = manager.get_progress_info()
        assert info["phase"] is None
        assert info["progress"] == 0
        assert info["total"] == 0

    def test_get_progress_info_active_phase(self):
        phases = ["validation"]
        manager = PhaseProgressManager(phases, use_tqdm=False)
        manager.start_phase("validation", 10)
        manager.update(3)
        info = manager.get_progress_info()
        assert info["phase"] == "validation"
        assert info["progress"] == 3
        assert info["total"] == 10
        assert info["percent"] == 30.0

    def test_set_description(self):
        phases = ["validation"]
        manager = PhaseProgressManager(phases, use_tqdm=False)
        # Without a tqdm progress bar, set_description should not crash
        manager.set_description("new desc")

    def test_use_tqdm_false_overrides(self):
        phases = ["validation"]
        manager = PhaseProgressManager(phases, use_tqdm=True)
        # If TQDM_AVAILABLE is False, use_tqdm should be False
        with mock.patch("labelme_tools.progress_logger.TQDM_AVAILABLE", False):
            manager2 = PhaseProgressManager(phases, use_tqdm=True)
            assert manager2.use_tqdm is False


class TestPrintPhaseHeader:

    def test_output_contains_phase_info(self, capsys):
        print_phase_header("validation", 0, 3)
        captured = capsys.readouterr()
        assert "阶段 1/3" in captured.out
        assert "数据验证" in captured.out

    def test_unknown_phase_name(self, capsys):
        print_phase_header("custom_step", 2, 5)
        captured = capsys.readouterr()
        assert "阶段 3/5" in captured.out
        assert "custom_step" in captured.out


class TestPrintPhaseFooter:

    def test_output_contains_completion_info(self, capsys):
        print_phase_footer("validation", 5, 10)
        captured = capsys.readouterr()
        assert "数据验证" in captured.out
        assert "5/10" in captured.out
        assert "50.0%" in captured.out

    def test_zero_total(self, capsys):
        print_phase_footer("validation", 0, 0)
        captured = capsys.readouterr()
        assert "100.0%" in captured.out

    def test_unknown_phase_name(self, capsys):
        print_phase_footer("custom_step", 3, 3)
        captured = capsys.readouterr()
        assert "custom_step" in captured.out


class TestIN_NOTEBOOK:

    def test_is_boolean(self):
        assert isinstance(IN_NOTEBOOK, bool)

    def test_tqdm_available_is_boolean(self):
        assert isinstance(TQDM_AVAILABLE, bool)