"""
测试 labelme_stats_processor 模块的核心功能
覆盖 FilterCopyResult, StatisticsFileProcessor, process_statistics_file
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from unsloth_finetune.data.labelme.labelme_stats_processor import (
    FilterCopyResult,
    StatisticsFileProcessor,
    process_statistics_file,
)


class TestFilterCopyResult:

    def test_default_values(self):
        result = FilterCopyResult(
            statistics_file="/path/to/stats.json",
            source_dir="/path/to/source",
            target_dir="/path/to/target",
        )
        assert result.statistics_file == "/path/to/stats.json"
        assert result.source_dir == "/path/to/source"
        assert result.target_dir == "/path/to/target"
        assert result.total_files_in_stats == 0
        assert result.copied_files == 0
        assert result.skipped_files == 0
        assert result.missing_files == []
        assert result.copied_file_paths == []
        assert result.labels_processed == []
        assert result.start_time is None
        assert result.end_time is None

    def test_duration_property(self):
        result = FilterCopyResult(
            statistics_file="/path/to/stats.json",
            source_dir="/path/to/source",
            target_dir="/path/to/target",
        )
        result.start_time = datetime(2024, 1, 1, 10, 0, 0)
        result.end_time = datetime(2024, 1, 1, 10, 0, 15)
        assert result.duration == 15.0

    def test_duration_none_when_no_times(self):
        result = FilterCopyResult(
            statistics_file="/path/to/stats.json",
            source_dir="/path/to/source",
            target_dir="/path/to/target",
        )
        assert result.duration is None

    def test_copy_ratio_property(self):
        result = FilterCopyResult(
            statistics_file="/path/to/stats.json",
            source_dir="/path/to/source",
            target_dir="/path/to/target",
            total_files_in_stats=10,
            copied_files=8,
        )
        assert result.copy_ratio == 80.0

    def test_copy_ratio_zero_total(self):
        result = FilterCopyResult(
            statistics_file="/path/to/stats.json",
            source_dir="/path/to/source",
            target_dir="/path/to/target",
            total_files_in_stats=0,
            copied_files=0,
        )
        assert result.copy_ratio is None

    def test_to_dict(self):
        result = FilterCopyResult(
            statistics_file="/path/to/stats.json",
            source_dir="/path/to/source",
            target_dir="/path/to/target",
            total_files_in_stats=5,
            copied_files=3,
            skipped_files=2,
        )
        d = result.to_dict()
        assert d["statistics_file"] == "/path/to/stats.json"
        assert d["total_files_in_stats"] == 5
        assert d["copied_files"] == 3
        assert d["skipped_files"] == 2
        assert d["copy_ratio"] == 60.0


class TestStatisticsFileProcessorInternal:

    @pytest.fixture
    def stats_file(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        for i in range(3):
            json_path = source_dir / f"img_{i:03d}.json"
            json_path.write_text(
                json.dumps({
                    "imagePath": f"img_{i:03d}.jpg",
                    "shapes": [{"label": "cat"}],
                }),
                encoding="utf-8",
            )
            (source_dir / f"img_{i:03d}.jpg").write_bytes(
                b"\xff\xd8\xff\xe0" + b"\x00" * 200
            )

        stats_path = tmp_path / "statistics.json"
        stats_data = {
            "source_dir": str(source_dir),
            "label_counts": {
                "cat": {"1": [f"img_{i:03d}.json" for i in range(3)]},
            },
        }
        stats_path.write_text(json.dumps(stats_data), encoding="utf-8")
        return stats_path, source_dir

    def test_load_statistics_file(self, stats_file):
        stats_path, _ = stats_file
        processor = StatisticsFileProcessor(
            statistics_file=str(stats_path),
            target_dir=str(stats_path.parent / "target"),
            use_tqdm=False,
        )
        data = processor._load_statistics_file()
        assert data is not None
        assert "label_counts" in data

    def test_load_statistics_file_not_exists(self, tmp_path):
        processor = StatisticsFileProcessor(
            statistics_file=str(tmp_path / "nonexistent.json"),
            target_dir=str(tmp_path / "target"),
            use_tqdm=False,
        )
        data = processor._load_statistics_file()
        assert data is None

    def test_extract_json_files(self, stats_file):
        stats_path, _ = stats_file
        processor = StatisticsFileProcessor(
            statistics_file=str(stats_path),
            target_dir=str(stats_path.parent / "target"),
            use_tqdm=False,
        )
        data = processor._load_statistics_file()
        files = processor._extract_json_files(data)
        assert len(files) == 3

    def test_extract_json_files_empty(self, tmp_path):
        processor = StatisticsFileProcessor(
            statistics_file=str(tmp_path / "stats.json"),
            target_dir=str(tmp_path / "target"),
            use_tqdm=False,
        )
        data = {"label_counts": {}}
        files = processor._extract_json_files(data)
        assert len(files) == 0

    def test_get_source_dir(self, stats_file):
        stats_path, source_dir = stats_file
        processor = StatisticsFileProcessor(
            statistics_file=str(stats_path),
            target_dir=str(stats_path.parent / "target"),
            use_tqdm=False,
        )
        data = processor._load_statistics_file()
        found_source = processor._get_source_dir(data)
        assert found_source is not None
        assert str(found_source) == str(source_dir)

    def test_get_source_dir_from_statistics_info(self, tmp_path):
        stats_path = tmp_path / "stats.json"
        stats_data = {
            "statistics_info": {"source_dir": str(tmp_path / "source")},
            "label_counts": {},
        }
        stats_path.write_text(json.dumps(stats_data), encoding="utf-8")
        processor = StatisticsFileProcessor(
            statistics_file=str(stats_path),
            target_dir=str(tmp_path / "target"),
            use_tqdm=False,
        )
        data = processor._load_statistics_file()
        found_source = processor._get_source_dir(data)
        assert found_source is not None

    def test_get_source_dir_none(self, tmp_path):
        stats_path = tmp_path / "stats.json"
        stats_data = {"label_counts": {}}
        stats_path.write_text(json.dumps(stats_data), encoding="utf-8")
        processor = StatisticsFileProcessor(
            statistics_file=str(stats_path),
            target_dir=str(tmp_path / "target"),
            use_tqdm=False,
        )
        data = processor._load_statistics_file()
        found_source = processor._get_source_dir(data)
        assert found_source is None

    def test_copy_file(self, stats_file, tmp_path):
        stats_path, source_dir = stats_file
        target_dir = tmp_path / "target"
        processor = StatisticsFileProcessor(
            statistics_file=str(stats_path),
            target_dir=str(target_dir),
            use_tqdm=False,
        )
        source_file = source_dir / "img_000.json"
        result = processor._copy_file(source_file, target_dir)
        assert result is not None
        assert Path(result).exists()


class TestStatisticsFileProcessorProcess:

    @pytest.fixture
    def full_setup(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        for i in range(3):
            json_path = source_dir / f"img_{i:03d}.json"
            json_path.write_text(
                json.dumps({
                    "imagePath": f"img_{i:03d}.jpg",
                    "shapes": [{"label": "cat"}],
                }),
                encoding="utf-8",
            )
            (source_dir / f"img_{i:03d}.jpg").write_bytes(
                b"\xff\xd8\xff\xe0" + b"\x00" * 200
            )

        stats_path = tmp_path / "statistics.json"
        stats_data = {
            "source_dir": str(source_dir),
            "label_counts": {
                "cat": {"1": [f"img_{i:03d}.json" for i in range(3)]},
            },
        }
        stats_path.write_text(json.dumps(stats_data), encoding="utf-8")

        target_dir = tmp_path / "target"
        return stats_path, source_dir, target_dir

    def test_process_basic(self, full_setup):
        stats_path, source_dir, target_dir = full_setup
        processor = StatisticsFileProcessor(
            statistics_file=str(stats_path),
            target_dir=str(target_dir),
            use_tqdm=False,
            copy_images=True,
        )
        result = processor.process()
        assert result.total_files_in_stats == 3
        assert result.copied_files == 3
        assert result.skipped_files == 0

    def test_process_missing_files(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        (source_dir / "img_000.json").write_text(
            json.dumps({"shapes": [{"label": "cat"}]}), encoding="utf-8"
        )

        stats_path = tmp_path / "statistics.json"
        stats_data = {
            "source_dir": str(source_dir),
            "label_counts": {
                "cat": {"1": ["img_000.json", "img_001.json", "img_002.json"]},
            },
        }
        stats_path.write_text(json.dumps(stats_data), encoding="utf-8")

        target_dir = tmp_path / "target"
        processor = StatisticsFileProcessor(
            statistics_file=str(stats_path),
            target_dir=str(target_dir),
            use_tqdm=False,
        )
        result = processor.process()
        assert result.total_files_in_stats == 3
        assert result.copied_files == 1
        assert result.skipped_files == 2
        assert len(result.missing_files) == 2

    def test_process_no_images(self, full_setup):
        stats_path, source_dir, target_dir = full_setup
        processor = StatisticsFileProcessor(
            statistics_file=str(stats_path),
            target_dir=str(target_dir),
            use_tqdm=False,
            copy_images=False,
        )
        result = processor.process()
        assert result.copied_files == 3

    def test_process_invalid_statistics(self, tmp_path):
        stats_path = tmp_path / "bad_stats.json"
        stats_path.write_text("{invalid json}", encoding="utf-8")

        target_dir = tmp_path / "target"
        processor = StatisticsFileProcessor(
            statistics_file=str(stats_path),
            target_dir=str(target_dir),
            use_tqdm=False,
        )
        result = processor.process()
        assert result.copied_files == 0


class TestProcessStatisticsFile:

    def test_function_returns_result(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "img.json").write_text(
            json.dumps({"shapes": [{"label": "cat"}]}), encoding="utf-8"
        )

        stats_path = tmp_path / "statistics.json"
        stats_data = {
            "source_dir": str(source_dir),
            "label_counts": {"cat": {"1": ["img.json"]}},
        }
        stats_path.write_text(json.dumps(stats_data), encoding="utf-8")

        target_dir = tmp_path / "target"
        result = process_statistics_file(
            statistics_file=str(stats_path),
            target_dir=str(target_dir),
            use_tqdm=False,
        )
        assert isinstance(result, FilterCopyResult)
        assert result.copied_files == 1
