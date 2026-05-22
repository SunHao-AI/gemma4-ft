"""
测试 labelme_converter 模块的核心功能
覆盖 BoundingBox, ConversionRecord, DatasetSplit, ConversionResult, LabelMeConverter
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from unsloth_finetune.data.labelme.labelme_converter import (
    BoundingBox,
    ConversionRecord,
    ConversionResult,
    DatasetSplit,
    LabelMeConverter,
    convert_to_unsloth_format,
)


# ============================================================
# BoundingBox 测试
# ============================================================

class TestBoundingBox:

    def test_basic_creation(self):
        bbox = BoundingBox(x_min=10.0, y_min=20.0, x_max=100.0, y_max=200.0, label="cat")
        assert bbox.x_min == 10.0
        assert bbox.y_min == 20.0
        assert bbox.x_max == 100.0
        assert bbox.y_max == 200.0
        assert bbox.label == "cat"

    def test_normalized_values(self):
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.8, label="dog")
        assert bbox.x_min < 1.0
        assert bbox.y_max < 1.0

    def test_zero_values(self):
        bbox = BoundingBox(x_min=0, y_min=0, x_max=0, y_max=0, label="empty")
        assert bbox.x_min == 0


# ============================================================
# ConversionRecord 测试
# ============================================================

class TestConversionRecord:

    def test_basic_creation(self):
        record = ConversionRecord(
            messages=[{"role": "user", "content": "test"}],
            images=["/path/to/image.jpg"],
            metadata={"json_path": "/path/to/file.json", "num_objects": 2},
            json_path="/path/to/file.json",
            image_path="/path/to/image.jpg",
        )
        assert len(record.messages) == 1
        assert len(record.images) == 1
        assert record.metadata["num_objects"] == 2

    def test_to_dict(self):
        record = ConversionRecord(
            messages=[{"role": "user", "content": "hello"}],
            images=["/path/img.jpg"],
            metadata={"key": "value"},
            json_path="/path/file.json",
            image_path="/path/img.jpg",
        )
        d = record.to_dict()
        assert "messages" in d
        assert "images" in d
        assert "metadata" in d
        assert d["messages"] == [{"role": "user", "content": "hello"}]
        assert d["metadata"]["key"] == "value"


# ============================================================
# DatasetSplit 测试
# ============================================================

class TestDatasetSplit:

    def test_default_values(self):
        split = DatasetSplit(split_name="train")
        assert split.split_name == "train"
        assert split.records == []
        assert split.total_records == 0
        assert split.total_images == 0
        assert split.total_objects == 0
        assert split.label_distribution == {}
        assert split.output_path is None

    def test_duration_property(self):
        split = DatasetSplit(split_name="train")
        split.start_time = datetime(2024, 1, 1, 10, 0, 0)
        split.end_time = datetime(2024, 1, 1, 10, 0, 10)
        assert split.duration == 10.0

    def test_duration_none(self):
        split = DatasetSplit(split_name="train")
        assert split.duration is None

    def test_to_dict(self):
        split = DatasetSplit(
            split_name="val",
            total_records=5,
            total_images=5,
            total_objects=10,
            label_distribution={"cat": 5, "dog": 5},
            output_path="/path/valid.jsonl",
        )
        d = split.to_dict()
        assert d["split_name"] == "val"
        assert d["total_records"] == 5
        assert d["total_images"] == 5
        assert d["total_objects"] == 10
        assert d["label_distribution"]["cat"] == 5
        assert d["output_path"] == "/path/valid.jsonl"


# ============================================================
# ConversionResult 测试
# ============================================================

class TestConversionResult:

    def test_default_values(self):
        result = ConversionResult(
            source_dir="/path/to/source",
            output_dir="/path/to/output",
        )
        assert result.source_dir == "/path/to/source"
        assert result.output_dir == "/path/to/output"
        assert result.train_split is None
        assert result.val_split is None
        assert result.test_split is None
        assert result.total_json_files == 0
        assert result.converted_count == 0
        assert result.failed_count == 0
        assert result.skipped_count == 0
        assert result.failed_files == []

    def test_duration_property(self):
        result = ConversionResult(
            source_dir="/path/to/source",
            output_dir="/path/to/output",
        )
        result.start_time = datetime(2024, 1, 1)
        result.end_time = datetime(2024, 1, 1, 0, 0, 30)
        assert result.duration == 30.0

    def test_conversion_rate_property(self):
        result = ConversionResult(
            source_dir="/path/to/source",
            output_dir="/path/to/output",
            total_json_files=100,
            converted_count=80,
        )
        assert result.conversion_rate == 80.0

    def test_conversion_rate_zero_total(self):
        result = ConversionResult(
            source_dir="/path/to/source",
            output_dir="/path/to/output",
            total_json_files=0,
            converted_count=0,
        )
        assert result.conversion_rate is None

    def test_to_dict(self):
        result = ConversionResult(
            source_dir="/path/to/source",
            output_dir="/path/to/output",
            total_json_files=10,
            converted_count=8,
            failed_count=1,
            skipped_count=1,
            instruction_text="test instruction",
        )
        d = result.to_dict()
        assert d["source_dir"] == "/path/to/source"
        assert d["total_json_files"] == 10
        assert d["converted_count"] == 8
        assert d["conversion_rate"] == 80.0
        assert d["instruction_text"] == "test instruction"


# ============================================================
# LabelMeConverter 内部方法测试
# ============================================================

class TestLabelMeConverterInternal:

    @pytest.fixture
    def converter(self, tmp_path):
        return LabelMeConverter(
            source_dir=str(tmp_path),
            output_dir=str(tmp_path / "output"),
            use_tqdm=False,
            validate_images=False,
        )

    def test_polygon_to_bbox(self, converter):
        points = [[10, 20], [100, 20], [100, 200], [10, 200]]
        bbox = converter._polygon_to_bbox(points, "cat")
        assert bbox.x_min == 10
        assert bbox.y_min == 20
        assert bbox.x_max == 100
        assert bbox.y_max == 200
        assert bbox.label == "cat"

    def test_polygon_to_bbox_invalid_points(self, converter):
        with pytest.raises(ValueError):
            converter._polygon_to_bbox([], "cat")

    def test_polygon_to_bbox_single_point(self, converter):
        with pytest.raises(ValueError):
            converter._polygon_to_bbox([[10, 20]], "cat")

    def test_rectangle_to_bbox(self, converter):
        shape = {
            "label": "dog",
            "points": [[10, 20], [100, 200]],
        }
        bbox = converter._rectangle_to_bbox(shape)
        assert bbox.x_min == 10
        assert bbox.y_min == 20
        assert bbox.x_max == 100
        assert bbox.y_max == 200
        assert bbox.label == "dog"

    def test_rectangle_to_bbox_swapped_coords(self, converter):
        shape = {
            "label": "cat",
            "points": [[100, 200], [10, 20]],
        }
        bbox = converter._rectangle_to_bbox(shape)
        assert bbox.x_min == 10
        assert bbox.x_max == 100

    def test_circle_to_bbox(self, converter):
        shape = {
            "label": "bird",
            "points": [[50, 50]],
            "radius": 25,
        }
        bbox = converter._circle_to_bbox(shape)
        assert bbox.x_min == 25
        assert bbox.y_min == 25
        assert bbox.x_max == 75
        assert bbox.y_max == 75

    def test_shape_to_bbox_polygon(self, converter):
        shape = {
            "label": "cat",
            "shape_type": "polygon",
            "points": [[10, 20], [100, 200], [50, 150]],
        }
        bbox = converter._shape_to_bbox(shape)
        assert bbox is not None
        assert bbox.label == "cat"

    def test_shape_to_bbox_rectangle(self, converter):
        shape = {
            "label": "dog",
            "shape_type": "rectangle",
            "points": [[10, 20], [100, 200]],
        }
        bbox = converter._shape_to_bbox(shape)
        assert bbox is not None

    def test_shape_to_bbox_circle(self, converter):
        shape = {
            "label": "bird",
            "shape_type": "circle",
            "points": [[50, 50]],
            "radius": 30,
        }
        bbox = converter._shape_to_bbox(shape)
        assert bbox is not None

    def test_shape_to_bbox_invalid(self, converter):
        shape = {
            "label": "invalid",
            "shape_type": "polygon",
            "points": [],
        }
        bbox = converter._shape_to_bbox(shape)
        # Should return None because of ValueError
        assert bbox is None

    def test_normalize_bbox(self, converter):
        bbox = BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200, label="cat")
        normalized = converter._normalize_bbox(bbox, 1000, 500)
        assert normalized.x_min == 0.01
        assert normalized.y_min == 0.04
        assert normalized.x_max == 0.1
        assert normalized.y_max == 0.4

    def test_normalize_bbox_invalid_dimensions(self, converter):
        bbox = BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200, label="cat")
        with pytest.raises(ValueError):
            converter._normalize_bbox(bbox, 0, 500)

    def test_format_bbox_string_normalized(self, converter):
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.8, label="cat")
        result = converter._format_bbox_string(bbox, normalized=True)
        assert "[0.1000, 0.2000, 0.5000, 0.8000]" == result

    def test_format_bbox_string_unnormalized(self, converter):
        bbox = BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200, label="cat")
        result = converter._format_bbox_string(bbox, normalized=False)
        assert "[10, 20, 100, 200]" == result

    def test_generate_conversation_messages(self, converter):
        bboxes = [
            BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.8, label="cat"),
            BoundingBox(x_min=0.3, y_min=0.1, x_max=0.7, y_max=0.6, label="dog"),
        ]
        messages = converter._generate_conversation_messages(bboxes)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        # JSONL格式: user content仅包含text，图片路径存储在images字段
        user_content = messages[0]["content"]
        assert len(user_content) == 1
        assert user_content[0]["type"] == "text"
        # 不应包含 {"type": "image"} 占位符
        assert not any(item.get("type") == "image" for item in user_content)

    def test_generate_category_specific_messages(self, converter):
        bboxes = [
            BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.8, label="cat"),
            BoundingBox(x_min=0.3, y_min=0.1, x_max=0.7, y_max=0.6, label="dog"),
        ]
        messages = converter._generate_category_specific_messages(bboxes, "cat")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        # JSONL格式: user content仅包含text
        user_content = messages[0]["content"]
        assert user_content[0]["type"] == "text"
        assert not any(item.get("type") == "image" for item in user_content)

    def test_split_dataset(self, converter):
        records = [
            ConversionRecord(
                messages=[],
                images=[f"/path/{i}.jpg"],
                metadata={"num_objects": 1},
                json_path=f"/path/{i}.json",
                image_path=f"/path/{i}.jpg",
            )
            for i in range(10)
        ]
        train, val, test = converter._split_dataset(records)
        assert len(train) == 8
        assert len(val) == 1
        assert len(test) == 1

    def test_split_dataset_keeps_same_image_in_single_split(self, converter):
        records = [
            ConversionRecord(
                messages=[],
                images=["/path/shared.jpg"],
                metadata={"num_objects": 1, "labels": ["cat"]},
                json_path="/path/shared.json",
                image_path="/path/shared.jpg",
            ),
            ConversionRecord(
                messages=[],
                images=["/path/shared.jpg"],
                metadata={"num_objects": 1, "labels": ["dog"]},
                json_path="/path/shared.json",
                image_path="/path/shared.jpg",
            ),
            ConversionRecord(
                messages=[],
                images=["/path/other.jpg"],
                metadata={"num_objects": 1, "labels": ["bird"]},
                json_path="/path/other.json",
                image_path="/path/other.jpg",
            ),
        ]

        train, val, test = converter._split_dataset(records)
        memberships = {}
        for split_name, split_records in (("train", train), ("val", val), ("test", test)):
            for record in split_records:
                memberships.setdefault(record.image_path, set()).add(split_name)

        assert len(memberships["/path/shared.jpg"]) == 1

    def test_save_split_counts_unique_images(self, converter, tmp_path):
        converter.output_dir = tmp_path
        records = [
            ConversionRecord(
                messages=[],
                images=["/path/shared.jpg"],
                metadata={"num_objects": 1, "labels": ["cat"]},
                json_path="/path/shared.json",
                image_path="/path/shared.jpg",
            ),
            ConversionRecord(
                messages=[],
                images=["/path/shared.jpg"],
                metadata={"num_objects": 2, "labels": ["dog"]},
                json_path="/path/shared.json",
                image_path="/path/shared.jpg",
            ),
        ]

        split = converter._save_split(records, "train")

        assert split.total_records == 2
        assert split.total_images == 1
        assert split.total_objects == 3


# ============================================================
# LabelMeConverter 转换流程测试
# ============================================================

class TestLabelMeConverterFlow:

    @pytest.fixture
    def converter_dir(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        for i in range(5):
            json_path = source_dir / f"img_{i:03d}.json"
            data = {
                "imagePath": f"img_{i:03d}.jpg",
                "imageWidth": 640,
                "imageHeight": 480,
                "shapes": [
                    {
                        "label": "cat",
                        "shape_type": "rectangle",
                        "points": [[10, 20], [100, 200]],
                    },
                ],
            }
            json_path.write_text(json.dumps(data), encoding="utf-8")
            (source_dir / f"img_{i:03d}.jpg").write_bytes(
                b"\xff\xd8\xff\xe0" + b"\x00" * 200
            )

        output_dir = tmp_path / "output"
        return source_dir, output_dir

    def test_convert_basic(self, converter_dir):
        source_dir, output_dir = converter_dir
        converter = LabelMeConverter(
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            use_tqdm=False,
            validate_images=False,
        )
        result = converter.convert()
        assert result.total_json_files == 5
        assert result.converted_count > 0
        assert result.train_split is not None
        assert result.train_split.output_path is not None
        assert (output_dir / "valid.jsonl").exists()

    def test_convert_with_validation(self, converter_dir):
        source_dir, output_dir = converter_dir
        converter = LabelMeConverter(
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            use_tqdm=False,
            validate_images=True,
        )
        result = converter.convert()
        assert result.total_json_files == 5

    def test_convert_no_json_files(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        output_dir = tmp_path / "output"
        converter = LabelMeConverter(
            source_dir=str(empty_dir),
            output_dir=str(output_dir),
            use_tqdm=False,
        )
        result = converter.convert()
        assert result.total_json_files == 0
        assert result.converted_count == 0

    def test_convert_per_category_mode(self, converter_dir):
        source_dir, output_dir = converter_dir
        converter = LabelMeConverter(
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            use_tqdm=False,
            validate_images=False,
            per_category_mode=True,
        )
        result = converter.convert()
        assert result.converted_count > 0

    def test_convert_with_selected_files(self, converter_dir):
        source_dir, output_dir = converter_dir
        selected = [str(source_dir / "img_000.json"), str(source_dir / "img_001.json")]
        converter = LabelMeConverter(
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            use_tqdm=False,
            validate_images=False,
            selected_files=selected,
        )
        result = converter.convert()
        assert result.total_json_files == 2

    def test_convert_no_normalize(self, converter_dir):
        source_dir, output_dir = converter_dir
        converter = LabelMeConverter(
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            use_tqdm=False,
            validate_images=False,
            normalize_coordinates=False,
        )
        result = converter.convert()
        assert result.converted_count > 0

    def test_convert_output_jsonl_exists(self, converter_dir):
        source_dir, output_dir = converter_dir
        converter = LabelMeConverter(
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            use_tqdm=False,
            validate_images=False,
        )
        result = converter.convert()
        if result.train_split and result.train_split.output_path:
            train_path = Path(result.train_split.output_path)
            assert train_path.exists()
            content = train_path.read_text(encoding="utf-8")
            lines = content.strip().split("\n")
            assert len(lines) > 0


# ============================================================
# convert_to_unsloth_format 便捷函数测试
# ============================================================

class TestConvertToUnslothFormat:

    def test_function_returns_result(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        json_path = source_dir / "img.json"
        json_path.write_text(
            json.dumps({
                "imagePath": "img.jpg",
                "imageWidth": 640,
                "imageHeight": 480,
                "shapes": [{"label": "cat", "shape_type": "rectangle", "points": [[10, 20], [100, 200]]}],
            }),
            encoding="utf-8",
        )

        output_dir = tmp_path / "output"
        result = convert_to_unsloth_format(
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            use_tqdm=False,
            validate_images=False,
        )
        assert isinstance(result, ConversionResult)
        assert result.total_json_files == 1
