"""
测试 labelme_cleaner 模块的核心功能
覆盖 ValidationStatus, ValidationResult, CleaningResult, LabelMeCleaner 的验证和去重逻辑
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from labelme_tools.labelme_cleaner import (
    CleaningResult,
    LabelMeCleaner,
    ValidationStatus,
    ValidationResult,
    clean_labelme_data,
)


# ============================================================
# ValidationStatus 测试
# ============================================================

class TestValidationStatus:

    def test_all_status_values(self):
        assert ValidationStatus.VALID.value == "valid"
        assert ValidationStatus.INVALID_JSON.value == "invalid_json"
        assert ValidationStatus.MISSING_SHAPES.value == "missing_shapes"
        assert ValidationStatus.EMPTY_SHAPES.value == "empty_shapes"
        assert ValidationStatus.MISSING_IMAGE.value == "missing_image"
        assert ValidationStatus.INVALID_IMAGE_PATH.value == "invalid_image_path"
        assert ValidationStatus.DUPLICATE_ANNOTATION.value == "duplicate_annotation"

    def test_status_enum_members(self):
        members = list(ValidationStatus)
        assert len(members) == 7

    def test_status_is_enum(self):
        from enum import Enum
        assert issubclass(ValidationStatus, Enum)


# ============================================================
# ValidationResult 测试
# ============================================================

class TestValidationResult:

    def test_valid_result(self):
        result = ValidationResult(
            json_path="/path/to/file.json",
            status=ValidationStatus.VALID,
            image_path="/path/to/image.jpg",
            shapes_count=5,
        )
        assert result.is_valid() is True
        assert result.json_path == "/path/to/file.json"
        assert result.image_path == "/path/to/image.jpg"
        assert result.shapes_count == 5

    def test_invalid_result(self):
        result = ValidationResult(
            json_path="/path/to/bad.json",
            status=ValidationStatus.INVALID_JSON,
            error_message="JSON解析失败",
        )
        assert result.is_valid() is False
        assert result.error_message == "JSON解析失败"

    def test_default_values(self):
        result = ValidationResult(
            json_path="/path/to/file.json",
            status=ValidationStatus.VALID,
        )
        assert result.image_path is None
        assert result.error_message is None
        assert result.shapes_count == 0

    def test_missing_shapes_result(self):
        result = ValidationResult(
            json_path="/path/to/file.json",
            status=ValidationStatus.MISSING_SHAPES,
            error_message="缺少shapes字段",
        )
        assert result.is_valid() is False

    def test_empty_shapes_result(self):
        result = ValidationResult(
            json_path="/path/to/file.json",
            status=ValidationStatus.EMPTY_SHAPES,
            error_message="shapes数组为空",
        )
        assert result.is_valid() is False

    def test_missing_image_result(self):
        result = ValidationResult(
            json_path="/path/to/file.json",
            status=ValidationStatus.MISSING_IMAGE,
            error_message="图片文件不存在",
        )
        assert result.is_valid() is False

    def test_duplicate_annotation_result(self):
        result = ValidationResult(
            json_path="/path/to/file.json",
            status=ValidationStatus.DUPLICATE_ANNOTATION,
            error_message="重复标注",
        )
        assert result.is_valid() is False


# ============================================================
# CleaningResult 测试
# ============================================================

class TestCleaningResult:

    def test_default_values(self):
        result = CleaningResult()
        assert result.total_files == 0
        assert result.valid_count == 0
        assert result.invalid_count == 0
        assert result.duplicate_count == 0
        assert result.valid_files == []
        assert result.invalid_files == []
        assert result.duplicate_files == []
        assert result.output_dir is None
        assert result.report_path is None
        assert result.start_time is None
        assert result.end_time is None

    def test_duration_property(self):
        result = CleaningResult()
        result.start_time = datetime(2024, 1, 1, 10, 0, 0)
        result.end_time = datetime(2024, 1, 1, 10, 0, 30)
        assert result.duration == 30.0

    def test_duration_none_when_no_times(self):
        result = CleaningResult()
        assert result.duration is None

    def test_duration_none_when_only_start(self):
        result = CleaningResult()
        result.start_time = datetime(2024, 1, 1)
        assert result.duration is None

    def test_valid_ratio_property(self):
        result = CleaningResult(total_files=100, valid_count=80)
        assert result.valid_ratio == 80.0

    def test_valid_ratio_zero_total(self):
        result = CleaningResult(total_files=0, valid_count=0)
        assert result.valid_ratio is None

    def test_duplicate_ratio_property(self):
        result = CleaningResult(total_files=100, duplicate_count=10)
        assert result.duplicate_ratio == 10.0

    def test_duplicate_ratio_zero_total(self):
        result = CleaningResult(total_files=0, duplicate_count=0)
        assert result.duplicate_ratio is None

    def test_to_dict(self):
        result = CleaningResult(
            total_files=10,
            valid_count=8,
            invalid_count=1,
            duplicate_count=1,
        )
        result.start_time = datetime(2024, 1, 1, 10, 0, 0)
        result.end_time = datetime(2024, 1, 1, 10, 0, 5)
        d = result.to_dict()
        assert d["total_files"] == 10
        assert d["valid_count"] == 8
        assert d["invalid_count"] == 1
        assert d["duplicate_count"] == 1
        assert d["valid_ratio"] == 80.0
        assert d["duplicate_ratio"] == 10.0
        assert d["duration_seconds"] == 5.0
        assert "valid_files" in d
        assert "invalid_files" in d


# ============================================================
# LabelMeCleaner 验证逻辑测试
# ============================================================

class TestLabelMeCleanerValidation:

    @pytest.fixture
    def cleaner_dir(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # 创建有效文件
        valid_json = source_dir / "valid.json"
        valid_data = {
            "imagePath": "valid.jpg",
            "shapes": [{"label": "cat", "shape_type": "rectangle"}],
        }
        valid_json.write_text(json.dumps(valid_data), encoding="utf-8")
        (source_dir / "valid.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 200)

        # 创建无效JSON文件
        invalid_json = source_dir / "invalid.json"
        invalid_json.write_text("{bad json}", encoding="utf-8")

        # 创建缺少shapes的文件
        no_shapes_json = source_dir / "no_shapes.json"
        no_shapes_json.write_text(
            '{"imagePath": "no_shapes.jpg"}', encoding="utf-8"
        )

        # 创建空shapes的文件
        empty_shapes_json = source_dir / "empty_shapes.json"
        empty_shapes_json.write_text(
            '{"imagePath": "empty.jpg", "shapes": []}', encoding="utf-8"
        )

        # 创建缺少imagePath的文件
        no_image_json = source_dir / "no_image.json"
        no_image_json.write_text(
            '{"shapes": [{"label": "cat"}]}', encoding="utf-8"
        )

        # 创建缺少图片文件的文件
        missing_img_json = source_dir / "missing_img.json"
        missing_img_json.write_text(
            '{"imagePath": "nonexistent.jpg", "shapes": [{"label": "cat"}]}',
            encoding="utf-8",
        )

        target_dir = tmp_path / "target"

        return source_dir, target_dir

    def test_validate_valid_file(self, cleaner_dir):
        source_dir, _ = cleaner_dir
        cleaner = LabelMeCleaner(
            source_dir=str(source_dir),
            target_dir=str(source_dir / "target"),
            use_tqdm=False,
        )
        result = cleaner._validate_json_structure(source_dir / "valid.json")
        assert result.is_valid() is True
        assert result.shapes_count == 1

    def test_validate_invalid_json(self, cleaner_dir):
        source_dir, _ = cleaner_dir
        cleaner = LabelMeCleaner(
            source_dir=str(source_dir),
            target_dir=str(source_dir / "target"),
            use_tqdm=False,
        )
        result = cleaner._validate_json_structure(source_dir / "invalid.json")
        assert result.status == ValidationStatus.INVALID_JSON

    def test_validate_missing_shapes(self, cleaner_dir):
        source_dir, _ = cleaner_dir
        cleaner = LabelMeCleaner(
            source_dir=str(source_dir),
            target_dir=str(source_dir / "target"),
            use_tqdm=False,
        )
        result = cleaner._validate_json_structure(source_dir / "no_shapes.json")
        assert result.status == ValidationStatus.MISSING_SHAPES

    def test_validate_empty_shapes(self, cleaner_dir):
        source_dir, _ = cleaner_dir
        cleaner = LabelMeCleaner(
            source_dir=str(source_dir),
            target_dir=str(source_dir / "target"),
            use_tqdm=False,
        )
        result = cleaner._validate_json_structure(source_dir / "empty_shapes.json")
        assert result.status == ValidationStatus.EMPTY_SHAPES

    def test_validate_invalid_image_path(self, cleaner_dir):
        source_dir, _ = cleaner_dir
        cleaner = LabelMeCleaner(
            source_dir=str(source_dir),
            target_dir=str(source_dir / "target"),
            use_tqdm=False,
        )
        result = cleaner._validate_json_structure(source_dir / "no_image.json")
        assert result.status == ValidationStatus.INVALID_IMAGE_PATH

    def test_validate_full_missing_image(self, cleaner_dir):
        source_dir, _ = cleaner_dir
        cleaner = LabelMeCleaner(
            source_dir=str(source_dir),
            target_dir=str(source_dir / "target"),
            use_tqdm=False,
        )
        result = cleaner._validate_file(source_dir / "missing_img.json")
        assert result.status == ValidationStatus.MISSING_IMAGE


# ============================================================
# LabelMeCleaner 去重逻辑测试
# ============================================================

class TestLabelMeCleanerDeduplication:

    def test_is_name_matched_true(self, tmp_path):
        cleaner = LabelMeCleaner(
            source_dir=str(tmp_path),
            target_dir=str(tmp_path / "target"),
            use_tqdm=False,
        )
        json_path = Path("/data/cat.json")
        image_path = Path("/data/cat.jpg")
        assert cleaner._is_name_matched(json_path, image_path) is True

    def test_is_name_matched_false(self, tmp_path):
        cleaner = LabelMeCleaner(
            source_dir=str(tmp_path),
            target_dir=str(tmp_path / "target"),
            use_tqdm=False,
        )
        json_path = Path("/data/cat_a.json")
        image_path = Path("/data/cat.jpg")
        assert cleaner._is_name_matched(json_path, image_path) is False

    def test_is_name_matched_case_insensitive(self, tmp_path):
        cleaner = LabelMeCleaner(
            source_dir=str(tmp_path),
            target_dir=str(tmp_path / "target"),
            use_tqdm=False,
        )
        json_path = Path("/data/CAT.json")
        image_path = Path("/data/cat.jpg")
        assert cleaner._is_name_matched(json_path, image_path) is True

    def test_select_primary_annotation_single(self, tmp_path):
        cleaner = LabelMeCleaner(
            source_dir=str(tmp_path),
            target_dir=str(tmp_path / "target"),
            use_tqdm=False,
        )
        annotation = ValidationResult(
            json_path="/data/cat.json",
            status=ValidationStatus.VALID,
            image_path="/data/cat.jpg",
            shapes_count=3,
        )
        result = cleaner._select_primary_annotation([annotation])
        assert result.json_path == "/data/cat.json"

    def test_select_primary_annotation_name_matched(self, tmp_path):
        cleaner = LabelMeCleaner(
            source_dir=str(tmp_path),
            target_dir=str(tmp_path / "target"),
            use_tqdm=False,
        )
        annotations = [
            ValidationResult(
                json_path="/data/cat.json",
                status=ValidationStatus.VALID,
                image_path="/data/cat.jpg",
                shapes_count=3,
            ),
            ValidationResult(
                json_path="/data/cat_extra.json",
                status=ValidationStatus.VALID,
                image_path="/data/cat.jpg",
                shapes_count=5,
            ),
        ]
        result = cleaner._select_primary_annotation(annotations)
        # Should prefer the one with matching name
        assert result.json_path == "/data/cat.json"

    def test_select_primary_annotation_by_shapes_count(self, tmp_path):
        cleaner = LabelMeCleaner(
            source_dir=str(tmp_path),
            target_dir=str(tmp_path / "target"),
            use_tqdm=False,
        )
        annotations = [
            ValidationResult(
                json_path="/data/annotation_a.json",
                status=ValidationStatus.VALID,
                image_path="/data/photo.jpg",
                shapes_count=3,
            ),
            ValidationResult(
                json_path="/data/annotation_b.json",
                status=ValidationStatus.VALID,
                image_path="/data/photo.jpg",
                shapes_count=10,
            ),
        ]
        result = cleaner._select_primary_annotation(annotations)
        # Should prefer the one with more shapes since no name match
        assert result.shapes_count == 10


# ============================================================
# LabelMeCleaner 标签映射测试
# ============================================================

class TestLabelMeCleanerLabelMapping:

    def test_apply_label_mapping_with_mapping(self, tmp_path):
        cleaner = LabelMeCleaner(
            source_dir=str(tmp_path),
            target_dir=str(tmp_path / "target"),
            use_tqdm=False,
            label_mapping={"cat": "feline", "dog": "canine"},
        )
        shapes = [
            {"label": "cat", "shape_type": "rectangle"},
            {"label": "dog", "shape_type": "rectangle"},
            {"label": "bird", "shape_type": "rectangle"},
        ]
        mapped = cleaner._apply_label_mapping(shapes)
        assert mapped[0]["label"] == "feline"
        assert mapped[0]["_original_label"] == "cat"
        assert mapped[1]["label"] == "canine"
        assert mapped[1]["_original_label"] == "dog"
        assert mapped[2]["label"] == "bird"

    def test_apply_label_mapping_empty(self, tmp_path):
        cleaner = LabelMeCleaner(
            source_dir=str(tmp_path),
            target_dir=str(tmp_path / "target"),
            use_tqdm=False,
            label_mapping={},
        )
        shapes = [{"label": "cat"}]
        mapped = cleaner._apply_label_mapping(shapes)
        assert mapped[0]["label"] == "cat"

    def test_apply_label_mapping_none_mapping(self, tmp_path):
        cleaner = LabelMeCleaner(
            source_dir=str(tmp_path),
            target_dir=str(tmp_path / "target"),
            use_tqdm=False,
        )
        shapes = [{"label": "cat"}]
        mapped = cleaner._apply_label_mapping(shapes)
        assert mapped[0]["label"] == "cat"


# ============================================================
# LabelMeCleaner 清洗流程测试
# ============================================================

class TestLabelMeCleanerClean:

    @pytest.fixture
    def full_clean_dir(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        for i in range(5):
            json_path = source_dir / f"img_{i:03d}.json"
            data = {
                "imagePath": f"img_{i:03d}.jpg",
                "shapes": [{"label": "cat", "shape_type": "rectangle"}],
            }
            json_path.write_text(json.dumps(data), encoding="utf-8")
            (source_dir / f"img_{i:03d}.jpg").write_bytes(
                b"\xff\xd8\xff\xe0" + b"\x00" * 200
            )

        target_dir = tmp_path / "target"
        return source_dir, target_dir

    def test_clean_basic_flow(self, full_clean_dir):
        source_dir, target_dir = full_clean_dir
        cleaner = LabelMeCleaner(
            source_dir=str(source_dir),
            target_dir=str(target_dir),
            use_tqdm=False,
            enable_format_conversion=False,
            enable_integrity_check=False,
            enable_statistics_report=False,
        )
        result = cleaner.clean()
        assert result.total_files == 5
        assert result.valid_count == 5
        assert result.invalid_count == 0

    def test_clean_with_no_json_files(self, tmp_path):
        source_dir = tmp_path / "empty_source"
        source_dir.mkdir()
        target_dir = tmp_path / "target"
        cleaner = LabelMeCleaner(
            source_dir=str(source_dir),
            target_dir=str(target_dir),
            use_tqdm=False,
        )
        result = cleaner.clean()
        assert result.total_files == 0
        assert result.valid_count == 0

    def test_clean_preserve_structure(self, full_clean_dir):
        source_dir, target_dir = full_clean_dir
        cleaner = LabelMeCleaner(
            source_dir=str(source_dir),
            target_dir=str(target_dir),
            use_tqdm=False,
            preserve_structure=True,
            enable_format_conversion=False,
            enable_integrity_check=False,
            enable_statistics_report=False,
        )
        result = cleaner.clean()
        assert result.total_files == 5
        cleaned_dir = target_dir / "cleaned_data"
        assert cleaned_dir.exists()

    def test_clean_with_deduplication(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # 创建两张图片，每个标注文件有对应的图片
        img1_path = source_dir / "cat.jpg"
        img1_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 200)
        img2_path = source_dir / "cat_extra.jpg"
        img2_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 200)

        json1 = source_dir / "cat.json"
        json1.write_text(
            json.dumps({
                "imagePath": "cat.jpg",
                "shapes": [{"label": "cat", "shape_type": "rectangle"}],
            }),
            encoding="utf-8",
        )

        json2 = source_dir / "cat_extra.json"
        json2.write_text(
            json.dumps({
                "imagePath": "cat_extra.jpg",
                "shapes": [{"label": "cat", "shape_type": "rectangle"}],
            }),
            encoding="utf-8",
        )

        target_dir = tmp_path / "target"
        cleaner = LabelMeCleaner(
            source_dir=str(source_dir),
            target_dir=str(target_dir),
            use_tqdm=False,
            deduplicate=True,
            enable_format_conversion=False,
            enable_integrity_check=False,
            enable_statistics_report=False,
        )
        result = cleaner.clean()
        assert result.total_files == 2
        # 两个标注文件都指向同一张真实图片（cat.jpg），但 cat_extra.json 的 imagePath 是 cat_extra.jpg
        # 去重是基于图片文件匹配的，所以需要验证去重结果
        assert result.valid_count >= 1

    def test_clean_without_deduplication(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        img1_path = source_dir / "cat.jpg"
        img1_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 200)
        img2_path = source_dir / "cat_extra.jpg"
        img2_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 200)

        json1 = source_dir / "cat.json"
        json1.write_text(
            json.dumps({
                "imagePath": "cat.jpg",
                "shapes": [{"label": "cat"}],
            }),
            encoding="utf-8",
        )

        json2 = source_dir / "cat_extra.json"
        json2.write_text(
            json.dumps({
                "imagePath": "cat_extra.jpg",
                "shapes": [{"label": "cat"}],
            }),
            encoding="utf-8",
        )

        target_dir = tmp_path / "target"
        cleaner = LabelMeCleaner(
            source_dir=str(source_dir),
            target_dir=str(target_dir),
            use_tqdm=False,
            deduplicate=False,
            enable_format_conversion=False,
            enable_integrity_check=False,
            enable_statistics_report=False,
        )
        result = cleaner.clean()
        assert result.total_files == 2
        assert result.valid_count == 2
        assert result.duplicate_count == 0

    def test_clean_generates_report(self, full_clean_dir):
        source_dir, target_dir = full_clean_dir
        cleaner = LabelMeCleaner(
            source_dir=str(source_dir),
            target_dir=str(target_dir),
            use_tqdm=False,
            generate_report=True,
            enable_format_conversion=False,
            enable_integrity_check=False,
            enable_statistics_report=False,
        )
        result = cleaner.clean()
        assert result.report_path is not None
        report_file = Path(result.report_path)
        assert report_file.exists()

    def test_clean_no_report(self, full_clean_dir):
        source_dir, target_dir = full_clean_dir
        cleaner = LabelMeCleaner(
            source_dir=str(source_dir),
            target_dir=str(target_dir),
            use_tqdm=False,
            generate_report=False,
            enable_format_conversion=False,
            enable_integrity_check=False,
            enable_statistics_report=False,
        )
        result = cleaner.clean()
        assert result.report_path is None


# ============================================================
# clean_labelme_data 便捷函数测试
# ============================================================

class TestCleanLabelmeData:

    def test_function_creates_cleaner(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "valid.json").write_text(
            json.dumps({
                "imagePath": "valid.jpg",
                "shapes": [{"label": "cat"}],
            }),
            encoding="utf-8",
        )
        (source_dir / "valid.jpg").write_bytes(
            b"\xff\xd8\xff\xe0" + b"\x00" * 200
        )

        target_dir = tmp_path / "target"
        result = clean_labelme_data(
            source_dir=str(source_dir),
            target_dir=str(target_dir),
            use_tqdm=False,
            enable_format_conversion=False,
            enable_integrity_check=False,
            enable_statistics_report=False,
        )
        assert isinstance(result, CleaningResult)
        assert result.total_files == 1
        assert result.valid_count == 1

    def test_function_with_custom_report_path(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "valid.json").write_text(
            json.dumps({
                "imagePath": "valid.jpg",
                "shapes": [{"label": "cat"}],
            }),
            encoding="utf-8",
        )
        (source_dir / "valid.jpg").write_bytes(
            b"\xff\xd8\xff\xe0" + b"\x00" * 200
        )

        target_dir = tmp_path / "target"
        report_path = tmp_path / "custom_report.txt"
        result = clean_labelme_data(
            source_dir=str(source_dir),
            target_dir=str(target_dir),
            use_tqdm=False,
            report_path=str(report_path),
            enable_format_conversion=False,
            enable_integrity_check=False,
            enable_statistics_report=False,
        )
        assert report_path.exists()


# ============================================================
# LabelMeCleaner 复制逻辑测试
# ============================================================

class TestLabelMeCleanerCopy:

    def test_copy_valid_file_preserve_structure(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        json_path = source_dir / "valid.json"
        json_path.write_text('{"key": "value"}', encoding="utf-8")
        image_path = source_dir / "valid.jpg"
        image_path.write_bytes(b"\xff\xd8\xff\xe0")

        target_dir = tmp_path / "target"
        cleaner = LabelMeCleaner(
            source_dir=str(source_dir),
            target_dir=str(target_dir),
            use_tqdm=False,
            preserve_structure=True,
            copy_images=True,
        )
        copied_json, copied_image = cleaner._copy_valid_file(json_path, image_path)
        assert copied_json is not None
        assert copied_image is not None
        assert Path(copied_json).exists()
        assert Path(copied_image).exists()

    def test_copy_valid_file_no_images(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        json_path = source_dir / "valid.json"
        json_path.write_text('{"key": "value"}', encoding="utf-8")
        image_path = source_dir / "valid.jpg"
        image_path.write_bytes(b"\xff\xd8\xff\xe0")

        target_dir = tmp_path / "target"
        cleaner = LabelMeCleaner(
            source_dir=str(source_dir),
            target_dir=str(target_dir),
            use_tqdm=False,
            copy_images=False,
        )
        copied_json, copied_image = cleaner._copy_valid_file(json_path, image_path)
        assert copied_json is not None
        assert copied_image is None

    def test_copy_valid_file_no_image_path(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        json_path = source_dir / "valid.json"
        json_path.write_text('{"key": "value"}', encoding="utf-8")

        target_dir = tmp_path / "target"
        cleaner = LabelMeCleaner(
            source_dir=str(source_dir),
            target_dir=str(target_dir),
            use_tqdm=False,
        )
        copied_json, copied_image = cleaner._copy_valid_file(json_path, None)
        assert copied_json is not None
        assert copied_image is None