"""Unified detection format specification and utilities.

Provides a consistent detection format from training through inference,
resolving the format mismatch that causes finetuned models to lose
bounding box output capability.

Two format modes:
- LABELME_TEXT: legacy Chinese free-text with [x_min, y_min, x_max, y_max]
- BOX_2D_JSON: JSON array with box_2d keys, normalized [x_min, y_min, x_max, y_max]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class OutputFormat(str, Enum):
    """Supported output format modes for labelme conversion."""
    LABELME_TEXT = "labelme_text"
    BOX_2D_JSON = "box_2d_json"


@dataclass
class DetectionFormatSpec:
    """Specification for a detection format."""
    name: str
    coordinate_order: str        # "xyxy" or "yxxy"
    coordinate_scale: str        # "normalized" or "1000x1000"
    response_structure: str      # "json_array" or "free_text"
    confidence_included: bool


FORMAT_SPECS: Dict[OutputFormat, DetectionFormatSpec] = {
    OutputFormat.LABELME_TEXT: DetectionFormatSpec(
        name="labelme_text",
        coordinate_order="xyxy",
        coordinate_scale="normalized",
        response_structure="free_text",
        confidence_included=False,
    ),
    OutputFormat.BOX_2D_JSON: DetectionFormatSpec(
        name="box_2d_json",
        coordinate_order="xyxy",
        coordinate_scale="normalized",
        response_structure="json_array",
        confidence_included=True,
    ),
}

DetectionPromptBuilder = Callable[[str], str]


def build_box_2d_json_response(bboxes: List[Dict[str, float]]) -> str:
    """Build a box_2d_json format response string from bbox list.

    Each bbox dict should have: x_min, y_min, x_max, y_max, label.
    Returns a JSON array string with box_2d keys and normalized coords.
    """
    detections = []
    for bbox in bboxes:
        detections.append({
            "box_2d": [
                round(bbox["x_min"], 4),
                round(bbox["y_min"], 4),
                round(bbox["x_max"], 4),
                round(bbox["y_max"], 4),
            ],
            "label": bbox["label"],
            "confidence": 1.0,
        })
    return json.dumps(detections, ensure_ascii=False)


def build_cn_normalized_detection_prompt(query: str) -> str:
    """Build Chinese normalized xyxy detection prompt.

    Coordinates: [x_min, y_min, x_max, y_max] normalized 0-1.
    """
    return (
        f"请仔细分析这张图片，{query}。\n"
        "\n"
        "如果检测到目标，请严格按照以下JSON格式返回检测结果:\n"
        '[\n  {"box_2d": [x_min, y_min, x_max, y_max], "label": "目标类别", "confidence": 1.0}\n]\n'
        "\n"
        "坐标说明:\n"
        "- box_2d: 归一化坐标 [x_min, y_min, x_max, y_max]，取值范围 [0, 1]\n"
        "- x_min < x_max, y_min < y_max\n"
        "- confidence: 置信度分数 (0.0-1.0)\n"
        "\n"
        "如果未检测到目标，请返回空数组: []"
    )


def build_en_normalized_detection_prompt(query: str) -> str:
    """Build English normalized xyxy detection prompt.

    Coordinates: [x_min, y_min, x_max, y_max] normalized 0-1.
    """
    return (
        f"Analyze this image carefully. {query}\n"
        "\n"
        "If the target is present, return only a JSON array with this schema:\n"
        '[\n  {"box_2d": [x_min, y_min, x_max, y_max], "label": "target", "confidence": 1.0}\n]\n'
        "\n"
        "Coordinate rules:\n"
        "- box_2d uses normalized [x_min, y_min, x_max, y_max] coordinates (0-1 range)\n"
        "- x_min < x_max, y_min < y_max\n"
        "- confidence must be between 0.0 and 1.0\n"
        "\n"
        "If no target is found, return []"
    )


def _extract_json_array(text: str) -> Optional[str]:
    """Extract a JSON array string from response text.

    First tries ```json code block, then balanced bracket extraction.
    """
    json_block = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if json_block:
        return json_block.group(1).strip()

    start_idx = text.find("[")
    if start_idx == -1:
        return None

    bracket_count = 0
    for i, char in enumerate(text[start_idx:], start_idx):
        if char == "[":
            bracket_count += 1
        elif char == "]":
            bracket_count -= 1
            if bracket_count == 0:
                return text[start_idx:i + 1]
    return None


def _is_normalized(coords: List[float]) -> bool:
    """Check if all coordinate values are in 0-1 normalized range."""
    return all(0 <= v <= 1 for v in coords)


def parse_box_2d_json_ground_truth(
    assistant_text: str,
    img_width: int,
    img_height: int,
    coord_order: str = "xyxy",
) -> List[Dict[str, Any]]:
    """Parse ground truth from box_2d_json format assistant text.

    Handles both normalized [x_min, y_min, x_max, y_max] (coord_order="xyxy")
    and legacy [y1, x1, y2, x2] (coord_order="yxxy") in box_2d.

    Returns list of dicts with pixel bbox: [x1, y1, x2, y2], label, confidence.
    """
    detections: List[Dict[str, Any]] = []

    def _convert(coords: List[float]) -> Tuple[int, int, int, int]:
        if coord_order == "xyxy":
            if _is_normalized(coords):
                return (
                    int(coords[0] * img_width),
                    int(coords[1] * img_height),
                    int(coords[2] * img_width),
                    int(coords[3] * img_height),
                )
            return int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        else:  # yxxy legacy
            scale_x = img_width / 1000.0
            scale_y = img_height / 1000.0
            if _is_normalized(coords):
                return (
                    int(coords[1] * img_width),
                    int(coords[0] * img_height),
                    int(coords[3] * img_width),
                    int(coords[2] * img_height),
                )
            return (
                int(coords[1] * scale_x),
                int(coords[0] * scale_y),
                int(coords[3] * scale_x),
                int(coords[2] * scale_y),
            )

    # Try full JSON array extraction
    json_str = _extract_json_array(assistant_text)
    if json_str:
        try:
            json_data = json.loads(json_str)
            if isinstance(json_data, list):
                for item in json_data:
                    if not isinstance(item, dict):
                        continue
                    coords = item.get("box_2d")
                    if not isinstance(coords, list) or len(coords) != 4:
                        continue
                    try:
                        float_coords = [float(c) for c in coords]
                    except (TypeError, ValueError):
                        continue
                    x1, y1, x2, y2 = _convert(float_coords)
                    confidence = item.get("confidence", 1.0)
                    try:
                        confidence = max(0.0, min(float(confidence), 1.0))
                    except (TypeError, ValueError):
                        confidence = 1.0
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "label": item.get("label", "object"),
                        "confidence": confidence,
                    })
                if detections:
                    return detections
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: individual JSON objects containing box_2d
    obj_pattern = r'\{[^{}]*"box_2d"[^{}]*\}'
    for obj_str in re.findall(obj_pattern, assistant_text, re.DOTALL):
        try:
            obj = json.loads(obj_str)
            if not isinstance(obj, dict):
                continue
            coords = obj.get("box_2d")
            if not isinstance(coords, list) or len(coords) != 4:
                continue
            try:
                float_coords = [float(c) for c in coords]
            except (TypeError, ValueError):
                continue
            x1, y1, x2, y2 = _convert(float_coords)
            confidence = obj.get("confidence", 1.0)
            try:
                confidence = max(0.0, min(float(confidence), 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": obj.get("label", "object"),
                "confidence": confidence,
            })
        except (json.JSONDecodeError, TypeError):
            continue

    return detections