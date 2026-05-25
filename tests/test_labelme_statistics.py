"""
测试 labelme_statistics 模块的核心功能
覆盖 LabelStatistics, LabelMeLabelStatistics, statistics_labelme_labels
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from unsloth_finetune.data.labelme.labelme_statistics import (
    LabelMeLabelStatistics,
    LabelStatistics,
    statistics_labelme_labels,
)


# ============================================================
# LabelStatistics 测试
# ============================================================

class TestLabelStatistics:

    def test_default_values(self):
        stats = LabelStatistics(source_dir="/path/to/dir")
        assert stats.source_dir == "/path/to/dir"
        assert stats.total_json_files == 0
        assert stats.processed_files == 0
        assert stats.skipped_files == 0
        assert stats.skipped_no_imageurl == 0
        assert stats.skipped_parse_error == 0
        assert stats.label_counts == {}
        assert stats.start_time is None
        assert stats.end_time is None

    def test_duration_property(self):
        stats = LabelStatistics(source_dir="/path/to/dir")
        stats.start_time = datetime(2024, 1, 1, 10, 0, 0)
        stats.end_time = datetime(2024, 1, 1, 10, 1, 30)
        assert stats.duration == 90.0

    def test_duration_none_when_no_times(self):
        stats = LabelStatistics(source_dir="/path/to/dir")
        assert stats.duration is None

    def test_total_labels_property(self):
        stats = LabelStatistics(
            source_dir="/path/to/dir",
            label_counts={"cat": {"1": {"a.json"}}, "dog": {"2": {"b.json"}}},
        )
        assert stats.total_labels == 2

    def test_total_labels_empty(self):
        stats = LabelStatistics(source_dir="/path/to/dir")
        assert stats.total_labels == 0

    def test_total_label_instances(self):
        stats = LabelStatistics(
            source_dir="/path/to/dir",
            label_counts={
                "cat": {"3": {"a.json", "b.json"}},
                "dog": {"1": {"c.json"}},
            },
        )
        # cat: 3 * 2 files = 6, dog: 1 * 1 file = 1, total = 7
        assert stats.total_label_instances == 7

    def test_total_label_instances_empty(self):
        stats = LabelStatistics(source_dir="/path/to/dir")
        assert stats.total_label_instances == 0

    def test_get_label_summary(self):
        stats = LabelStatistics(
            source_dir="/path/to/dir",
            label_counts={
                "cat": {"3": {"a.json", "b.json"}, "1": {"c.json"}},
            },
        )
        summary = stats.get_label_summary()
        assert "cat" in summary
        assert summary["cat"]["total_files"] == 3
        assert summary["cat"]["total_instances"] == 7  # 3*2 + 1*1
        assert summary["cat"]["max_per_file"] == 3
        assert summary["cat"]["min_per_file"] == 1

    def test_to_dict(self):
        stats = LabelStatistics(
            source_dir="/path/to/dir",
            total_json_files=10,
            processed_files=8,
        )
        stats.start_time = datetime(2024, 1, 1)
        stats.end_time = datetime(2024, 1, 1, 0, 0, 5)
        d = stats.to_dict()
        assert d["source_dir"] == "/path/to/dir"
        assert d["total_json_files"] == 10
        assert d["processed_files"] == 8
        assert "label_counts" in d
        assert "label_summary" in d
        assert d["duration_seconds"] == 5.0

    def test_to_structured_dict_sorted(self):
        stats = LabelStatistics(
            source_dir="/path/to/dir",
            label_counts={
                "dog": {"2": {"b.json"}, "1": {"a.json"}},
                "cat": {"3": {"c.json"}, "1": {"d.json"}},
            },
        )
        structured = stats.to_structured_dict()
        labels = list(structured["label_counts"].keys())
        # Sorted alphabetically
        assert labels[0] == "cat"
        assert labels[1] == "dog"
        # Count keys sorted numerically
        cat_counts = list(structured["label_counts"]["cat"].keys())
        assert cat_counts[0] == "1"
        assert cat_counts[1] == "3"


# ============================================================
# LabelMeLabelStatistics 内部方法测试
# ============================================================

class TestLabelMeLabelStatisticsInternal:

    @pytest.fixture
    def stats_instance(self, tmp_path):
        return LabelMeLabelStatistics(
            source_dir=str(tmp_path),
            use_tqdm=False,
        )

    def test_get_file_path_str_relative(self, tmp_path):
        stats = LabelMeLabelStatistics(
            source_dir=str(tmp_path),
            use_relative_path=True,
            use_tqdm=False,
        )
        file_path = tmp_path / "subdir" / "file.json"
        file_path.parent.mkdir(parents=True)
        file_path.write_text("{}", encoding="utf-8")
        result = stats._get_file_path_str(file_path)
        assert "subdir" in result
        assert "file.json" in result

    def test_get_file_path_str_absolute(self, tmp_path):
        stats = LabelMeLabelStatistics(
            source_dir=str(tmp_path),
            use_relative_path=False,
            use_tqdm=False,
        )
        file_path = tmp_path / "file.json"
        file_path.write_text("{}", encoding="utf-8")
        result = stats._get_file_path_str(file_path)
        assert str(tmp_path) in result

    def test_has_image_reference_with_url(self, stats_instance):
        data = {"imageUrl": "https://example.com/image.jpg"}
        assert stats_instance._has_image_reference(data)

    def test_has_image_reference_with_path(self, stats_instance):
        data = {"imagePath": "image.jpg"}
        assert stats_instance._has_image_reference(data)

    def test_has_image_reference_with_both(self, stats_instance):
        data = {"imageUrl": "https://example.com/image.jpg", "imagePath": "image.jpg"}
        assert stats_instance._has_image_reference(data)

    def test_has_image_reference_without_both(self, stats_instance):
        data = {"otherField": "value"}
        assert stats_instance._has_image_reference(data) is False

    def test_has_image_reference_empty_url(self, stats_instance):
        data = {"imageUrl": ""}
        assert not stats_instance._has_image_reference(data)

    def test_has_image_reference_empty_path(self, stats_instance):
        data = {"imagePath": ""}
        assert not stats_instance._has_image_reference(data)

    def test_count_labels_in_file(self, stats_instance):
        data = {
            "shapes": [
                {"label": "cat"},
                {"label": "cat"},
                {"label": "dog"},
            ]
        }
        result = stats_instance._count_labels_in_file(data)
        assert result["cat"] == 2
        assert result["dog"] == 1

    def test_count_labels_no_shapes(self, stats_instance):
        data = {"shapes": []}
        result = stats_instance._count_labels_in_file(data)
        assert result == {}

    def test_count_labels_empty_label(self, stats_instance):
        data = {"shapes": [{"label": ""}, {"label": "cat"}]}
        result = stats_instance._count_labels_in_file(data)
        assert "cat" in result
        assert "" not in result

    def test_count_labels_no_label_key(self, stats_instance):
        data = {"shapes": [{"type": "rectangle"}]}
        result = stats_instance._count_labels_in_file(data)
        assert result == {}

    def test_count_labels_shapes_not_list(self, stats_instance):
        data = {"shapes": "invalid"}
        result = stats_instance._count_labels_in_file(data)
        assert result == {}

    def test_update_global_dict(self, stats_instance):
        global_dict = {}
        lock = threading.Lock()
        stats_instance._update_global_dict(
            global_dict, {"cat": 3}, "a.json", lock
        )
        assert "cat" in global_dict
        assert "3" in global_dict["cat"]
        assert "a.json" in global_dict["cat"]["3"]

    def test_update_global_dict_multiple_files(self, stats_instance):
        global_dict = {}
        lock = threading.Lock()
        stats_instance._update_global_dict(
            global_dict, {"cat": 2}, "a.json", lock
        )
        stats_instance._update_global_dict(
            global_dict, {"cat": 2}, "b.json", lock
        )
        stats_instance._update_global_dict(
            global_dict, {"cat": 1}, "c.json", lock
        )
        assert "2" in global_dict["cat"]
        assert "1" in global_dict["cat"]
        assert len(global_dict["cat"]["2"]) == 2


# ============================================================
# LabelMeLabelStatistics 统计流程测试
# ============================================================

class TestLabelMeLabelStatisticsFlow:

    @pytest.fixture
    def stats_dir(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        for i in range(5):
            json_path = source_dir / f"img_{i:03d}.json"
            label = "cat" if i < 3 else "dog"
            data = {
                "imageUrl": f"https://example.com/img_{i:03d}.jpg",
                "shapes": [
                    {"label": label, "shape_type": "rectangle"},
                    {"label": label, "shape_type": "rectangle"} if i < 2 else {},
                ],
            }
            # Remove empty shapes for clean data
            if i >= 2:
                data["shapes"] = [{"label": label, "shape_type": "rectangle"}]

            json_path.write_text(json.dumps(data), encoding="utf-8")

        return source_dir

    def test_statistics_basic(self, stats_dir):
        stats = LabelMeLabelStatistics(
            source_dir=str(stats_dir),
            use_tqdm=False,
            max_workers=1,
        )
        result = stats.statistics()
        assert result.total_json_files == 5
        assert result.processed_files > 0

    def test_statistics_no_json_files(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        stats = LabelMeLabelStatistics(
            source_dir=str(empty_dir),
            use_tqdm=False,
        )
        result = stats.statistics()
        assert result.total_json_files == 0
        assert result.processed_files == 0

    def test_statistics_skip_no_image_ref(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        json_path = source_dir / "no_ref.json"
        json_path.write_text(
            json.dumps({"shapes": [{"label": "cat"}]}),
            encoding="utf-8",
        )
        stats = LabelMeLabelStatistics(
            source_dir=str(source_dir),
            use_tqdm=False,
            max_workers=1,
        )
        result = stats.statistics()
        assert result.skipped_no_imageurl == 1
        assert result.processed_files == 0

    def test_statistics_imagepath_accepted(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        json_path = source_dir / "local_ref.json"
        json_path.write_text(
            json.dumps({"imagePath": "img.jpg", "shapes": [{"label": "cat"}]}),
            encoding="utf-8",
        )
        stats = LabelMeLabelStatistics(
            source_dir=str(source_dir),
            use_tqdm=False,
            max_workers=1,
        )
        result = stats.statistics()
        assert result.skipped_no_imageurl == 0
        assert result.processed_files == 1

    def test_statistics_skip_parse_error(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        json_path = source_dir / "bad.json"
        json_path.write_text("{invalid json}", encoding="utf-8")
        stats = LabelMeLabelStatistics(
            source_dir=str(source_dir),
            use_tqdm=False,
            max_workers=1,
        )
        result = stats.statistics()
        assert result.skipped_parse_error >= 1


# ============================================================
# statistics_labelme_labels 便捷函数测试
# ============================================================

class TestStatisticsLabelmeLabels:

    def test_function_returns_label_statistics(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        json_path = source_dir / "img.json"
        json_path.write_text(
            json.dumps({
                "imageUrl": "https://example.com/img.jpg",
                "shapes": [{"label": "cat"}],
            }),
            encoding="utf-8",
        )
        result = statistics_labelme_labels(
            source_dir=str(source_dir),
            use_tqdm=False,
            max_workers=1,
        )
        assert isinstance(result, LabelStatistics)
        assert result.total_json_files == 1

    def test_function_with_recursive(self, tmp_path):
        source_dir = tmp_path / "source"
        sub_dir = source_dir / "sub"
        sub_dir.mkdir(parents=True)
        json_path = sub_dir / "img.json"
        json_path.write_text(
            json.dumps({
                "imageUrl": "https://example.com/img.jpg",
                "shapes": [{"label": "cat"}],
            }),
            encoding="utf-8",
        )
        result = statistics_labelme_labels(
            source_dir=str(source_dir),
            recursive=True,
            use_tqdm=False,
            max_workers=1,
        )
        assert result.total_json_files == 1

    def test_function_non_recursive(self, tmp_path):
        source_dir = tmp_path / "source"
        sub_dir = source_dir / "sub"
        sub_dir.mkdir(parents=True)
        json_path = sub_dir / "img.json"
        json_path.write_text(
            json.dumps({
                "imageUrl": "https://example.com/img.jpg",
                "shapes": [{"label": "cat"}],
            }),
            encoding="utf-8",
        )
        # Root level also has a file
        root_json = source_dir / "root_img.json"
        root_json.write_text(
            json.dumps({
                "imageUrl": "https://example.com/root.jpg",
                "shapes": [{"label": "dog"}],
            }),
            encoding="utf-8",
        )
        result = statistics_labelme_labels(
            source_dir=str(source_dir),
            recursive=False,
            use_tqdm=False,
            max_workers=1,
        )
        assert result.total_json_files == 1
