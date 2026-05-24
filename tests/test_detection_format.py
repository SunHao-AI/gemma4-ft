"""Tests for detection_format module."""

import json

import pytest

from unsloth_finetune.data.labelme.detection_format import (
    DetectionFormatSpec,
    DetectionPromptBuilder,
    OutputFormat,
    FORMAT_SPECS,
    build_box_2d_json_response,
    build_cn_normalized_detection_prompt,
    build_en_normalized_detection_prompt,
    parse_box_2d_json_ground_truth,
    convert_xyxy_to_format,
    _extract_json_array,
    _is_normalized,
)


# ---------------------------------------------------------------------------
# 1. OutputFormat enum values
# ---------------------------------------------------------------------------

class TestOutputFormat:
    def test_labelme_text_value(self):
        assert OutputFormat.LABELME_TEXT.value == "labelme_text"

    def test_box_2d_json_value(self):
        assert OutputFormat.BOX_2D_JSON.value == "box_2d_json"

    def test_enum_is_str_enum(self):
        """OutputFormat members should behave like strings."""
        assert isinstance(OutputFormat.LABELME_TEXT, str)
        assert isinstance(OutputFormat.BOX_2D_JSON, str)

    def test_enum_members_count(self):
        assert len(OutputFormat) == 2

    def test_enum_from_value(self):
        assert OutputFormat("labelme_text") == OutputFormat.LABELME_TEXT
        assert OutputFormat("box_2d_json") == OutputFormat.BOX_2D_JSON


# ---------------------------------------------------------------------------
# 2. DetectionFormatSpec and FORMAT_SPECS completeness
# ---------------------------------------------------------------------------

class TestDetectionFormatSpec:
    def test_dataclass_fields(self):
        spec = DetectionFormatSpec(
            name="test",
            coordinate_format="xyxy",
            coordinate_scale="norm_1",
            response_structure="json_array",
            confidence_included=True,
        )
        assert spec.name == "test"
        assert spec.coordinate_format == "xyxy"
        assert spec.coordinate_scale == "norm_1"
        assert spec.response_structure == "json_array"
        assert spec.confidence_included is True

    def test_format_specs_contains_both_formats(self):
        assert OutputFormat.LABELME_TEXT in FORMAT_SPECS
        assert OutputFormat.BOX_2D_JSON in FORMAT_SPECS
        assert len(FORMAT_SPECS) == 2

    def test_labelme_text_spec_values(self):
        spec = FORMAT_SPECS[OutputFormat.LABELME_TEXT]
        assert spec.name == "labelme_text"
        assert spec.coordinate_format == "xyxy"
        assert spec.coordinate_scale == "norm_1"
        assert spec.response_structure == "free_text"
        assert spec.confidence_included is False

    def test_box_2d_json_spec_values(self):
        spec = FORMAT_SPECS[OutputFormat.BOX_2D_JSON]
        assert spec.name == "box_2d_json"
        assert spec.coordinate_format == "xyxy"
        assert spec.coordinate_scale == "norm_1"
        assert spec.response_structure == "json_array"
        assert spec.confidence_included is True

    def test_detection_prompt_builder_type_alias(self):
        """DetectionPromptBuilder should be a callable type alias."""
        # It's a type alias; just verify it's callable-compatible at the type level
        # by checking that functions with (str)->str signature are compatible.
        assert callable(build_cn_normalized_detection_prompt)
        assert callable(build_en_normalized_detection_prompt)


# ---------------------------------------------------------------------------
# 3. build_box_2d_json_response
# ---------------------------------------------------------------------------

class TestBuildBox2dJsonResponse:
    def test_single_bbox_produces_valid_json(self):
        bbox = {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.8, "label": "cat"}
        result = build_box_2d_json_response([bbox])
        parsed = json.loads(result)
        assert len(parsed) == 1
        det = parsed[0]
        assert "box_2d" in det
        assert "label" in det
        assert "confidence" in det
        assert det["box_2d"] == [0.1, 0.2, 0.5, 0.8]
        assert det["label"] == "cat"
        assert det["confidence"] == 1.0

    def test_multiple_bboxes_produces_valid_json_array(self):
        bboxes = [
            {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.8, "label": "cat"},
            {"x_min": 0.3, "y_min": 0.4, "x_max": 0.6, "y_max": 0.9, "label": "dog"},
        ]
        result = build_box_2d_json_response(bboxes)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["label"] == "cat"
        assert parsed[1]["label"] == "dog"

    def test_coordinates_rounded_to_4_decimal_places(self):
        bbox = {"x_min": 0.12345678, "y_min": 0.9999999, "x_max": 0.5555555, "y_max": 0.1111111, "label": "car"}
        result = build_box_2d_json_response([bbox])
        parsed = json.loads(result)
        box = parsed[0]["box_2d"]
        assert box[0] == round(0.12345678, 4)  # 0.1235
        assert box[1] == round(0.9999999, 4)    # 1.0
        assert box[2] == round(0.5555555, 4)    # 0.5556
        assert box[3] == round(0.1111111, 4)    # 0.1111

    def test_confidence_is_always_1(self):
        bboxes = [
            {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.8, "label": "cat"},
            {"x_min": 0.3, "y_min": 0.4, "x_max": 0.6, "y_max": 0.9, "label": "dog"},
        ]
        result = build_box_2d_json_response(bboxes)
        parsed = json.loads(result)
        for det in parsed:
            assert det["confidence"] == 1.0

    def test_label_is_preserved(self):
        bbox = {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.8, "label": "special_object_42"}
        result = build_box_2d_json_response([bbox])
        parsed = json.loads(result)
        assert parsed[0]["label"] == "special_object_42"

    def test_result_is_valid_json_string(self):
        bbox = {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.8, "label": "cat"}
        result = build_box_2d_json_response([bbox])
        assert isinstance(result, str)
        # Should parse without error
        json.loads(result)

    def test_ensure_ascii_false_preserves_unicode_labels(self):
        bbox = {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.8, "label": "汽车"}
        result = build_box_2d_json_response([bbox])
        assert "汽车" in result  # not escaped as \uXXXX


# ---------------------------------------------------------------------------
# 4. build_cn_normalized_detection_prompt
# ---------------------------------------------------------------------------

class TestBuildCnNormalizedDetectionPrompt:
    def test_contains_format_instructions(self):
        prompt = build_cn_normalized_detection_prompt("检测所有行人")
        assert "JSON格式" in prompt
        assert "box_2d" in prompt

    def test_query_is_embedded(self):
        prompt = build_cn_normalized_detection_prompt("检测所有行人")
        assert "检测所有行人" in prompt

    def test_mentions_box_2d(self):
        prompt = build_cn_normalized_detection_prompt("test")
        assert "box_2d" in prompt

    def test_mentions_xyxy_coordinate_format(self):
        prompt = build_cn_normalized_detection_prompt("test")
        assert "x_min, y_min, x_max, y_max" in prompt

    def test_mentions_normalized(self):
        prompt = build_cn_normalized_detection_prompt("test")
        assert "归一化" in prompt

    def test_mentions_confidence(self):
        prompt = build_cn_normalized_detection_prompt("test")
        assert "confidence" in prompt

    def test_mentions_empty_array(self):
        prompt = build_cn_normalized_detection_prompt("test")
        assert "[]" in prompt


# ---------------------------------------------------------------------------
# 5. build_en_normalized_detection_prompt
# ---------------------------------------------------------------------------

class TestBuildEnNormalizedDetectionPrompt:
    def test_contains_format_instructions(self):
        prompt = build_en_normalized_detection_prompt("Find all pedestrians")
        assert "JSON" in prompt
        assert "box_2d" in prompt

    def test_query_is_embedded(self):
        prompt = build_en_normalized_detection_prompt("Find all pedestrians")
        assert "Find all pedestrians" in prompt

    def test_mentions_box_2d(self):
        prompt = build_en_normalized_detection_prompt("test")
        assert "box_2d" in prompt

    def test_mentions_xyxy_coordinate_format(self):
        prompt = build_en_normalized_detection_prompt("test")
        assert "x_min, y_min, x_max, y_max" in prompt

    def test_mentions_normalized(self):
        prompt = build_en_normalized_detection_prompt("test")
        assert "normalized" in prompt

    def test_mentions_confidence(self):
        prompt = build_en_normalized_detection_prompt("test")
        assert "confidence" in prompt

    def test_mentions_empty_array(self):
        prompt = build_en_normalized_detection_prompt("test")
        assert "[]" in prompt


# ---------------------------------------------------------------------------
# 6. parse_box_2d_json_ground_truth
# ---------------------------------------------------------------------------

class TestParseBox2dJsonGroundTruth:
    # -- normalized xyxy conversion --

    def test_single_normalized_bbox_xyxy(self):
        """Normalized [0.1, 0.2, 0.5, 0.8] with 640x480 -> [64, 96, 320, 384]."""
        text = json.dumps([
            {"box_2d": [0.1, 0.2, 0.5, 0.8], "label": "cat", "confidence": 0.9}
        ])
        result = parse_box_2d_json_ground_truth(text, 640, 480)
        assert len(result) == 1
        assert result[0]["bbox"] == [64, 96, 320, 384]
        assert result[0]["label"] == "cat"
        assert result[0]["confidence"] == 0.9

    def test_multiple_bboxes_parsed(self):
        text = json.dumps([
            {"box_2d": [0.1, 0.2, 0.5, 0.8], "label": "cat", "confidence": 0.9},
            {"box_2d": [0.3, 0.4, 0.7, 0.6], "label": "dog", "confidence": 0.7},
        ])
        result = parse_box_2d_json_ground_truth(text, 640, 480)
        assert len(result) == 2
        assert result[0]["bbox"] == [64, 96, 320, 384]
        assert result[1]["bbox"] == [192, 192, 448, 288]

    def test_empty_array_returns_empty_list(self):
        result = parse_box_2d_json_ground_truth("[]", 640, 480)
        assert result == []

    def test_markdown_json_wrapped_response(self):
        text = (
            "Here is the result:\n"
            "```json\n"
            '[{"box_2d": [0.1, 0.2, 0.5, 0.8], "label": "cat", "confidence": 0.9}]\n'
            "```"
        )
        result = parse_box_2d_json_ground_truth(text, 640, 480)
        assert len(result) == 1
        assert result[0]["bbox"] == [64, 96, 320, 384]

    # -- malformed input --

    def test_malformed_json_returns_empty(self):
        result = parse_box_2d_json_ground_truth("not json at all", 640, 480)
        assert result == []

    def test_json_without_box_2d_returns_empty(self):
        text = json.dumps([{"label": "cat", "confidence": 0.9}])
        result = parse_box_2d_json_ground_truth(text, 640, 480)
        assert result == []

    def test_partial_valid_items_in_array(self):
        """Array with one valid and one invalid item should return only the valid one."""
        text = json.dumps([
            {"box_2d": [0.1, 0.2, 0.5, 0.8], "label": "cat", "confidence": 0.9},
            {"label": "invalid", "confidence": 0.5},  # no box_2d
        ])
        result = parse_box_2d_json_ground_truth(text, 640, 480)
        assert len(result) == 1
        assert result[0]["label"] == "cat"

    # -- individual JSON object fallback --

    def test_individual_json_object_fallback(self):
        """When no valid array can be parsed, loose JSON objects with box_2d are found."""
        text = (
            'I found a cat: {"box_2d": [0.1, 0.2, 0.5, 0.8], "label": "cat", "confidence": 0.9}'
        )
        result = parse_box_2d_json_ground_truth(text, 640, 480)
        assert len(result) == 1
        assert result[0]["bbox"] == [64, 96, 320, 384]
        assert result[0]["label"] == "cat"

    def test_multiple_loose_json_objects_fallback(self):
        text = (
            '{"box_2d": [0.1, 0.2, 0.5, 0.8], "label": "cat", "confidence": 0.9}\n'
            '{"box_2d": [0.3, 0.4, 0.7, 0.6], "label": "dog", "confidence": 0.7}'
        )
        result = parse_box_2d_json_ground_truth(text, 640, 480)
        assert len(result) == 2

    # -- pixel-scale coordinates (values > 1) --

    def test_pixel_scale_coords_xyxy(self):
        """Values > 1 are treated as pixel coords and kept as-is (modulo int cast)."""
        text = json.dumps([
            {"box_2d": [64, 96, 320, 384], "label": "cat", "confidence": 0.9}
        ])
        result = parse_box_2d_json_ground_truth(text, 640, 480)
        assert result[0]["bbox"] == [64, 96, 320, 384]

    # -- confidence defaulting --

    def test_confidence_defaults_to_1_when_missing(self):
        text = json.dumps([
            {"box_2d": [0.1, 0.2, 0.5, 0.8], "label": "cat"}
        ])
        result = parse_box_2d_json_ground_truth(text, 640, 480)
        assert result[0]["confidence"] == 1.0

    def test_confidence_clamped_to_max_1(self):
        text = json.dumps([
            {"box_2d": [0.1, 0.2, 0.5, 0.8], "label": "cat", "confidence": 2.5}
        ])
        result = parse_box_2d_json_ground_truth(text, 640, 480)
        assert result[0]["confidence"] == 1.0

    def test_confidence_clamped_to_min_0(self):
        text = json.dumps([
            {"box_2d": [0.1, 0.2, 0.5, 0.8], "label": "cat", "confidence": -0.5}
        ])
        result = parse_box_2d_json_ground_truth(text, 640, 480)
        assert result[0]["confidence"] == 0.0

    def test_confidence_invalid_type_defaults_to_1(self):
        text = json.dumps([
            {"box_2d": [0.1, 0.2, 0.5, 0.8], "label": "cat", "confidence": "high"}
        ])
        result = parse_box_2d_json_ground_truth(text, 640, 480)
        assert result[0]["confidence"] == 1.0

    # -- label defaulting --

    def test_label_defaults_to_object_when_missing(self):
        text = json.dumps([
            {"box_2d": [0.1, 0.2, 0.5, 0.8], "confidence": 0.9}
        ])
        result = parse_box_2d_json_ground_truth(text, 640, 480)
        assert result[0]["label"] == "object"

    # -- box_2d with wrong length --

    def test_box_2d_with_3_coords_skipped(self):
        text = json.dumps([
            {"box_2d": [0.1, 0.2, 0.5], "label": "cat", "confidence": 0.9}
        ])
        result = parse_box_2d_json_ground_truth(text, 640, 480)
        assert result == []

    def test_box_2d_with_5_coords_skipped(self):
        text = json.dumps([
            {"box_2d": [0.1, 0.2, 0.5, 0.8, 0.0], "label": "cat", "confidence": 0.9}
        ])
        result = parse_box_2d_json_ground_truth(text, 640, 480)
        assert result == []

    # -- non-numeric coords --

    def test_box_2d_with_string_coords_skipped(self):
        text = json.dumps([
            {"box_2d": ["a", "b", "c", "d"], "label": "cat", "confidence": 0.9}
        ])
        result = parse_box_2d_json_ground_truth(text, 640, 480)
        assert result == []


# ---------------------------------------------------------------------------
# 7. _extract_json_array helper
# ---------------------------------------------------------------------------

class TestExtractJsonArray:
    def test_plain_json_array(self):
        text = '[{"a": 1}]'
        assert _extract_json_array(text) == '[{"a": 1}]'

    def test_markdown_json_block(self):
        text = '```json\n[{"a": 1}]\n```'
        result = _extract_json_array(text)
        assert result == '[{"a": 1}]'

    def test_no_array_returns_none(self):
        text = "no array here"
        assert _extract_json_array(text) is None

    def test_embedded_array_in_text(self):
        text = 'result: [{"a": 1}] done'
        assert _extract_json_array(text) == '[{"a": 1}]'

    def test_nested_brackets_balanced(self):
        text = '[{"a": [1, 2]}]'
        assert _extract_json_array(text) == '[{"a": [1, 2]}]'


# ---------------------------------------------------------------------------
# 8. _is_normalized helper
# ---------------------------------------------------------------------------

class TestIsNormalized:
    def test_all_values_in_01_range(self):
        assert _is_normalized([0.0, 0.5, 1.0, 0.2]) is True

    def test_value_above_1(self):
        assert _is_normalized([0.1, 1.5, 0.5, 0.8]) is False

    def test_negative_value(self):
        assert _is_normalized([-0.1, 0.2, 0.5, 0.8]) is False

    def test_exact_boundary_values(self):
        assert _is_normalized([0.0, 1.0]) is True


# ---------------------------------------------------------------------------
# 9. convert_xyxy_to_format
# ---------------------------------------------------------------------------

class TestConvertXyxyToFormat:
    def test_xyxy_passthrough(self):
        result = convert_xyxy_to_format(10, 20, 100, 200, "xyxy")
        assert result == [10, 20, 100, 200]

    def test_xywh_conversion(self):
        result = convert_xyxy_to_format(10, 20, 100, 200, "xywh")
        assert result == [10, 20, 90, 180]

    def test_cxcywh_conversion(self):
        result = convert_xyxy_to_format(10, 20, 100, 200, "cxcywh")
        assert result == [55.0, 110.0, 90, 180]

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Unknown coord_format"):
            convert_xyxy_to_format(10, 20, 100, 200, "yxxy")

    def test_xyxy_is_default(self):
        result = convert_xyxy_to_format(10, 20, 100, 200)
        assert result == [10, 20, 100, 200]