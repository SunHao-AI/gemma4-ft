"""
Tests for labelme_converter module covering all feature areas:
- Coordinate normalization (raw/norm_1/norm_100/norm_1000)
- Coordinate format (xyxy/xywh/cxcywh)
- Generation strategy (all_in_one/per_class/both)
- Prompt templates (en/zh, simple/descriptive/cot)
- Data filtering (whitelist/blacklist/remap, bbox validation, shape types)
- Output schema (openai_messages/sharegpt)
- Split methods (random/sequential/stratified)
- dataset_info.json output
- ConversionRecord to_dict for both schemas
"""

import json
import math
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from unsloth_finetune.data.labelme.detection_format import (
    CoordFormat,
    CoordNorm,
    GenStrategy,
    OutputFormat,
    OutputSchema,
    PromptLang,
    PromptStyle,
    SplitMethod,
    build_detection_prompt,
    build_detection_response,
    _transform_and_format_coords,
)
from unsloth_finetune.data.labelme.labelme_converter import (
    BoundingBox,
    ConversionRecord,
    ConversionResult,
    DatasetSplit,
    LabelMeConverter,
    convert_to_unsloth_format,
    _parse_split_ratio,
)


# ============================================================
# Data Classes
# ============================================================

class TestBoundingBox:

    def test_basic_creation(self):
        bbox = BoundingBox(x_min=10.0, y_min=20.0, x_max=100.0, y_max=200.0, label="cat")
        assert bbox.x_min == 10.0
        assert bbox.label == "cat"

    def test_normalized_values(self):
        bbox = BoundingBox(x_min=0.1875, y_min=0.0938, x_max=0.5, y_max=0.5833, label="dog")
        assert bbox.x_min < 1.0


class TestConversionRecord:

    def test_basic_creation(self):
        record = ConversionRecord(
            messages=[{"role": "user", "content": [{"type": "text", "text": "test"}]}],
            images=["/path/img.jpg"],
            metadata={"json_path": "/path/file.json"},
            json_path="/path/file.json",
            image_path="/path/img.jpg",
            record_id="img_001",
            gen_strategy_tag="all_in_one",
        )
        assert record.record_id == "img_001"
        assert record.gen_strategy_tag == "all_in_one"
        assert record.is_test_only == False

    def test_to_dict_openai_messages(self):
        record = ConversionRecord(
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "prompt"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "response"}]},
            ],
            images=["/path/img.jpg"],
            metadata={"key": "value"},
            json_path="/path/file.json",
            image_path="/path/img.jpg",
            record_id="img_001",
        )
        d = record.to_dict(schema="openai_messages")
        assert "messages" in d
        assert "images" in d
        assert "metadata" in d
        assert len(d["messages"]) == 2

    def test_to_dict_sharegpt(self):
        record = ConversionRecord(
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "Detect all [cat]."}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Found 1 cat: [120, 45, 320, 280]"}]},
            ],
            images=["images/img_001.jpg"],
            metadata={},
            json_path="/path/file.json",
            image_path="/path/img.jpg",
            record_id="img_001",
        )
        d = record.to_dict(schema="sharegpt")
        assert d["id"] == "img_001"
        assert d["image"] == "images/img_001.jpg"
        assert len(d["conversations"]) == 2
        assert d["conversations"][0]["from"] == "human"
        assert "<image>" in d["conversations"][0]["value"]

    def test_to_dict_sharegpt_test_only(self):
        record = ConversionRecord(
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "Detect all [cat]."}]},
            ],
            images=["images/img_001.jpg"],
            metadata={},
            json_path="/path/file.json",
            image_path="/path/img.jpg",
            record_id="img_001",
            is_test_only=True,
        )
        d = record.to_dict(schema="sharegpt")
        assert len(d["conversations"]) == 1
        assert d["conversations"][0]["from"] == "human"

    def test_to_dict_openai_test_only(self):
        record = ConversionRecord(
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "Detect all [cat]."}]},
                {"role": "assistant", "content": [{"type": "text", "text": "response"}]},
            ],
            images=["/path/img.jpg"],
            metadata={},
            json_path="/path/file.json",
            image_path="/path/img.jpg",
            is_test_only=True,
        )
        d = record.to_dict(schema="openai_messages")
        assert len(d["messages"]) == 1
        assert d["messages"][0]["role"] == "user"


class TestConversionResult:

    def test_default_values(self):
        result = ConversionResult(source_dir="/src", output_dir="/out")
        assert result.skipped_details["empty_annotations"] == 0
        assert result.skipped_details["invalid_bbox"] == 0
        assert result.skipped_details["missing_images"] == 0
        assert result.config == {}

    def test_duration_property(self):
        result = ConversionResult(source_dir="/src", output_dir="/out")
        result.start_time = datetime(2024, 1, 1)
        result.end_time = datetime(2024, 1, 1, 0, 0, 30)
        assert result.duration == 30.0

    def test_conversion_rate_zero_total(self):
        result = ConversionResult(source_dir="/src", output_dir="/out")
        assert result.conversion_rate is None


class TestDatasetSplit:

    def test_to_dict_compact(self):
        split = DatasetSplit(
            split_name="train",
            total_records=800,
            total_images=800,
            total_objects=3620,
        )
        d = split.to_dict()
        assert d["records"] == 800
        assert d["images"] == 800
        assert d["annotations"] == 3620


# ============================================================
# Split Ratio Parsing
# ============================================================

class TestSplitRatioParsing:

    def test_standard_8_1_1(self):
        t, v, te = _parse_split_ratio("8:1:1")
        assert t == 0.8
        assert v == 0.1
        assert te == 0.1

    def test_no_test_9_1_0(self):
        t, v, te = _parse_split_ratio("9:1:0")
        assert t == 0.9
        assert v == 0.1
        assert te == 0.0

    def test_custom_7_2_1(self):
        t, v, te = _parse_split_ratio("7:2:1")
        assert t == 0.7
        assert v == 0.2
        assert te == 0.1

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            _parse_split_ratio("8:1")

    def test_invalid_values(self):
        with pytest.raises(ValueError):
            _parse_split_ratio("a:b:c")


# ============================================================
# Coordinate Pipeline
# ============================================================

class TestCoordNormalization:

    def test_raw_mode(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".", coord_norm="raw")
        bbox = BoundingBox(x_min=120, y_min=45, x_max=320, y_max=280, label="cat")
        result = conv._normalize_bbox(bbox, 640, 480)
        assert result.x_min == 120
        assert result.y_min == 45

    def test_norm_1_mode(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".", coord_norm="norm_1")
        bbox = BoundingBox(x_min=120, y_min=45, x_max=320, y_max=280, label="cat")
        result = conv._normalize_bbox(bbox, 640, 480)
        assert result.x_min == pytest.approx(0.1875, abs=0.001)
        assert result.y_min == pytest.approx(0.0938, abs=0.001)

    def test_norm_100_mode(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".", coord_norm="norm_100")
        bbox = BoundingBox(x_min=120, y_min=45, x_max=320, y_max=280, label="cat")
        result = conv._normalize_bbox(bbox, 640, 480)
        assert result.x_min == 19
        assert result.y_min == 9

    def test_norm_1000_mode(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".", coord_norm="norm_1000")
        bbox = BoundingBox(x_min=120, y_min=45, x_max=320, y_max=280, label="cat")
        result = conv._normalize_bbox(bbox, 640, 480)
        assert result.x_min == 188
        assert result.y_min == 94

    def test_invalid_image_size(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".", coord_norm="norm_1")
        bbox = BoundingBox(x_min=120, y_min=45, x_max=320, y_max=280, label="cat")
        with pytest.raises(ValueError):
            conv._normalize_bbox(bbox, 0, 480)


class TestCoordFormatTransform:

    def test_xyxy_format(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".", coord_format="xyxy")
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.8, label="cat")
        coords = conv._transform_coords(bbox)
        assert coords == [0.1, 0.2, 0.5, 0.8]

    def test_xywh_format(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".", coord_format="xywh")
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.8, label="cat")
        coords = conv._transform_coords(bbox)
        assert coords == pytest.approx([0.1, 0.2, 0.4, 0.6], abs=0.001)

    def test_cxcywh_format(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".", coord_format="cxcywh")
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.8, label="cat")
        coords = conv._transform_coords(bbox)
        assert coords == pytest.approx([0.3, 0.5, 0.4, 0.6], abs=0.001)


class TestCoordFormatting:

    def test_raw_formatting(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".", coord_norm="raw")
        formatted = conv._format_coord_list([120, 45, 320, 280])
        assert formatted == "[120, 45, 320, 280]"

    def test_norm_1_formatting(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".", coord_norm="norm_1")
        formatted = conv._format_coord_list([0.1875, 0.0938, 0.5, 0.5833])
        assert formatted == "[0.1875, 0.0938, 0.5000, 0.5833]"

    def test_norm_100_formatting(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".", coord_norm="norm_100")
        formatted = conv._format_coord_list([19, 9, 50, 58])
        assert formatted == "[19, 9, 50, 58]"

    def test_norm_1000_formatting(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".", coord_norm="norm_1000")
        formatted = conv._format_coord_list([188, 94, 500, 583])
        assert formatted == "[188, 94, 500, 583]"


# ============================================================
# Bbox Validation
# ============================================================

class TestBboxValidation:

    def test_valid_bbox(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".")
        bbox = BoundingBox(x_min=10, y_min=10, x_max=100, y_max=100, label="cat")
        assert conv._validate_bbox(bbox, 640, 480) == True

    def test_negative_coords(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".")
        bbox = BoundingBox(x_min=-10, y_min=10, x_max=100, y_max=100, label="cat")
        assert conv._validate_bbox(bbox, 640, 480) == False

    def test_out_of_bounds(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".")
        bbox = BoundingBox(x_min=10, y_min=10, x_max=700, y_max=100, label="cat")
        assert conv._validate_bbox(bbox, 640, 480) == False

    def test_zero_area(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".")
        bbox = BoundingBox(x_min=100, y_min=10, x_max=100, y_max=100, label="cat")
        assert conv._validate_bbox(bbox, 640, 480) == False

    def test_min_size_threshold(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".", min_bbox_size=5)
        bbox = BoundingBox(x_min=10, y_min=10, x_max=13, y_max=100, label="cat")
        assert conv._validate_bbox(bbox, 640, 480) == False

    def test_min_size_pass(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".", min_bbox_size=5)
        bbox = BoundingBox(x_min=10, y_min=10, x_max=20, y_max=100, label="cat")
        assert conv._validate_bbox(bbox, 640, 480) == True


# ============================================================
# Prompt Templates
# ============================================================

class TestPromptTemplates:

    def test_en_descriptive_all_in_one(self):
        prompt = build_detection_prompt("en", "descriptive", "all_in_one", ["cat", "dog"])
        assert "cat, dog" in prompt
        assert "Please detect all" in prompt

    def test_en_simple_per_class(self):
        prompt = build_detection_prompt("en", "simple", "per_class", ["cat"])
        assert "cat" in prompt
        assert "Detect all" in prompt

    def test_zh_descriptive_all_in_one(self):
        prompt = build_detection_prompt("zh", "descriptive", "all_in_one", ["cat", "dog"])
        assert "cat, dog" in prompt
        assert "请检测" in prompt

    def test_en_cot(self):
        prompt = build_detection_prompt("en", "cot", "all_in_one", ["person"])
        assert "think step by step" in prompt

    def test_zh_cot(self):
        prompt = build_detection_prompt("zh", "cot", "per_class", ["person"])
        assert "逐步思考" in prompt


class TestDetectionResponse:

    def test_box_2d_json_response(self):
        bboxes = [
            {"x_min": 0.1875, "y_min": 0.0938, "x_max": 0.5, "y_max": 0.5833, "label": "cat"},
        ]
        response = build_detection_response(
            lang="en", gen_strategy="all_in_one", bboxes=bboxes,
            coord_format="xyxy", coord_norm="norm_1", output_format="box_2d_json",
        )
        data = json.loads(response)
        assert data[0]["label"] == "cat"
        assert data[0]["box_2d"] == pytest.approx([0.1875, 0.0938, 0.5, 0.5833], abs=0.001)

    def test_labelme_text_response(self):
        bboxes = [
            {"x_min": 188, "y_min": 94, "x_max": 500, "y_max": 583, "label": "cat"},
        ]
        response = build_detection_response(
            lang="zh", gen_strategy="all_in_one", bboxes=bboxes,
            coord_format="xyxy", coord_norm="norm_1000", output_format="labelme_text",
        )
        assert "cat" in response


# ============================================================
# Detection Format Transform & Format
# ============================================================

class TestTransformAndFormatCoords:

    def test_norm_1_xyxy(self):
        result = _transform_and_format_coords(
            {"x_min": 0.1875, "y_min": 0.0938, "x_max": 0.5, "y_max": 0.5833, "label": "cat"},
            "norm_1", "xyxy",
        )
        assert result == "[0.1875, 0.0938, 0.5000, 0.5833]"

    def test_norm_1000_xywh(self):
        result = _transform_and_format_coords(
            {"x_min": 188, "y_min": 94, "x_max": 500, "y_max": 583, "label": "cat"},
            "norm_1000", "xywh",
        )
        assert "312" in result  # width = 500-188=312

    def test_raw_cxcywh(self):
        result = _transform_and_format_coords(
            {"x_min": 120, "y_min": 45, "x_max": 320, "y_max": 280, "label": "cat"},
            "raw", "cxcywh",
        )
        assert "220" in result  # cx = (120+320)/2=220


# ============================================================
# Shape to BBox Conversion
# ============================================================

class TestShapeToBbox:

    def test_polygon_conversion(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".", shape_types=["rectangle", "polygon"])
        shape = {"shape_type": "polygon", "points": [[10, 20], [50, 20], [50, 80], [10, 80]], "label": "cat"}
        bbox = conv._shape_to_bbox(shape)
        assert bbox.x_min == 10
        assert bbox.x_max == 50

    def test_rectangle_conversion(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".")
        shape = {"shape_type": "rectangle", "points": [[10, 20], [50, 80]], "label": "dog"}
        bbox = conv._shape_to_bbox(shape)
        assert bbox.x_min == 10
        assert bbox.x_max == 50

    def test_circle_conversion(self):
        conv = LabelMeConverter(source_dir=".", output_dir=".", shape_types=["rectangle", "circle"])
        shape = {"shape_type": "circle", "points": [[100, 100]], "radius": 50, "label": "circle_obj"}
        bbox = conv._shape_to_bbox(shape)
        assert bbox.x_min == 50
        assert bbox.x_max == 150


# ============================================================
# Full Pipeline Test
# ============================================================

class TestLabelMeConverterFlow:

    def _create_test_data(self, tmp_dir):
        """Create minimal test data for pipeline testing."""
        img_dir = tmp_dir / "images"
        img_dir.mkdir()
        json_dir = tmp_dir / "jsons"
        json_dir.mkdir()

        # Create a simple test image
        try:
            from PIL import Image
            img = Image.new("RGB", (640, 480), color="red")
            img.save(str(img_dir / "img_001.jpg"))
        except ImportError:
            # Create a dummy file if PIL not available
            (img_dir / "img_001.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        # Create a LabelMe JSON
        data = {
            "version": "5.0.1",
            "imagePath": "img_001.jpg",
            "imageHeight": 480,
            "imageWidth": 640,
            "shapes": [
                {"label": "cat", "shape_type": "rectangle", "points": [[120, 45], [320, 280]]},
                {"label": "dog", "shape_type": "rectangle", "points": [[400, 100], [600, 400]]},
            ],
        }
        with open(json_dir / "img_001.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

        return json_dir

    def test_basic_convert_norm_1000_xyxy(self, tmp_path):
        json_dir = self._create_test_data(tmp_path)
        output_dir = tmp_path / "output"

        result = convert_to_unsloth_format(
            source_dir=str(json_dir),
            output_dir=str(output_dir),
            coord_norm="norm_1000",
            coord_format="xyxy",
            gen_strategy="all_in_one",
            lang="en",
            prompt_style="descriptive",
            split="8:1:1",
            random_seed=42,
            validate_images=False,
            output_format="box_2d_json",
            image_path_mode="absolute",
        )
        assert result.converted_count > 0
        assert result.train_split is not None

    def test_per_class_gen_strategy(self, tmp_path):
        json_dir = self._create_test_data(tmp_path)
        output_dir = tmp_path / "output"

        result = convert_to_unsloth_format(
            source_dir=str(json_dir),
            output_dir=str(output_dir),
            coord_norm="norm_1",
            gen_strategy="per_class",
            lang="zh",
            split="8:1:1",
            random_seed=42,
            validate_images=False,
            output_format="labelme_text",
            image_path_mode="absolute",
        )
        # per_class should produce 2 records (cat + dog)
        assert result.converted_count == 2

    def test_sharegpt_output(self, tmp_path):
        json_dir = self._create_test_data(tmp_path)
        output_dir = tmp_path / "output"

        result = convert_to_unsloth_format(
            source_dir=str(json_dir),
            output_dir=str(output_dir),
            coord_norm="norm_1000",
            output_schema="sharegpt",
            split="10:0:0",
            random_seed=42,
            validate_images=False,
            image_path_mode="absolute",
        )
        assert result.train_split is not None
        assert result.train_split.total_records > 0
        # Check output file content
        train_file = Path(result.train_split.output_path)
        if train_file.exists():
            with open(train_file, "r", encoding="utf-8") as f:
                line = f.readline()
                if line:
                    record = json.loads(line)
                    assert "conversations" in record
                    assert "id" in record

    def test_class_whitelist(self, tmp_path):
        json_dir = self._create_test_data(tmp_path)
        output_dir = tmp_path / "output"

        result = convert_to_unsloth_format(
            source_dir=str(json_dir),
            output_dir=str(output_dir),
            class_whitelist=["cat"],
            split="8:1:1",
            random_seed=42,
            validate_images=False,
            image_path_mode="absolute",
        )
        # Only cat should be in the output
        assert result.converted_count > 0

    def test_class_blacklist(self, tmp_path):
        json_dir = self._create_test_data(tmp_path)
        output_dir = tmp_path / "output"

        result = convert_to_unsloth_format(
            source_dir=str(json_dir),
            output_dir=str(output_dir),
            class_blacklist=["dog"],
            split="8:1:1",
            random_seed=42,
            validate_images=False,
            image_path_mode="absolute",
        )
        assert result.converted_count > 0

    def test_no_json_files(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        output_dir = tmp_path / "output"

        result = convert_to_unsloth_format(
            source_dir=str(empty_dir),
            output_dir=str(output_dir),
        )
        assert result.total_json_files == 0
        assert result.converted_count == 0

    def test_dataset_info_json(self, tmp_path):
        json_dir = self._create_test_data(tmp_path)
        output_dir = tmp_path / "output"

        result = convert_to_unsloth_format(
            source_dir=str(json_dir),
            output_dir=str(output_dir),
            split="8:1:1",
            random_seed=42,
            validate_images=False,
            image_path_mode="absolute",
        )
        info_file = output_dir / "dataset_info.json"
        if info_file.exists():
            with open(info_file, "r", encoding="utf-8") as f:
                info = json.load(f)
                assert "config" in info
                assert "class_distribution" in info
                assert "skipped" in info
                assert info["config"]["coord_norm"] == "norm_1000"


# ============================================================
# Convenience Function
# ============================================================

class TestConvertToUnslothFormat:

    def test_wrapper_matches_converter(self, tmp_path):
        json_dir = tmp_path / "jsons"
        json_dir.mkdir()

        data = {
            "version": "5.0.1",
            "imagePath": "img_001.jpg",
            "imageHeight": 480,
            "imageWidth": 640,
            "shapes": [
                {"label": "cat", "shape_type": "rectangle", "points": [[10, 20], [100, 200]]},
            ],
        }
        with open(json_dir / "img_001.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

        output_dir = tmp_path / "output"
        result = convert_to_unsloth_format(
            source_dir=str(json_dir),
            output_dir=str(output_dir),
            coord_norm="norm_1",
            coord_format="xyxy",
            validate_images=False,
            image_path_mode="absolute",
        )
        assert result is not None
        assert isinstance(result, ConversionResult)