"""Unified detection format specification and utilities.

Provides a consistent detection format from training through inference,
resolving the format mismatch that causes finetuned models to lose
bounding box output capability.

Format modes:
- LABELME_TEXT: Chinese free-text with [x_min, y_min, x_max, y_max] normalized
- BOX_2D_JSON: JSON array with box_2d keys, normalized [x_min, y_min, x_max, y_max]

Coordinate formats (xyxy, xywh, cxcywh) are standard CV formats — no legacy
non-standard orderings.

Also provides a prompt template registry for generating detection prompts
in multiple languages and styles.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class OutputFormat(str, Enum):
    """Supported output format modes for labelme conversion."""
    LABELME_TEXT = "labelme_text"
    BOX_2D_JSON = "box_2d_json"


class CoordNorm(str, Enum):
    """Coordinate normalization modes."""
    RAW = "raw"
    NORM_1 = "norm_1"
    NORM_100 = "norm_100"
    NORM_1000 = "norm_1000"


class CoordFormat(str, Enum):
    """Coordinate output format modes."""
    XYXY = "xyxy"
    XYWH = "xywh"
    CXCYWH = "cxcywh"


def convert_xyxy_to_format(
    x1: float, y1: float, x2: float, y2: float,
    coord_format: str = "xyxy",
) -> List[float]:
    """Convert xyxy pixel coordinates to the requested output format.

    Args:
        x1, y1, x2, y2: pixel coordinates in xyxy format
        coord_format: "xyxy", "yxyx", "xywh", or "cxcywh"

    Returns:
        List of 4 floats in the requested format.

    Note:
        - xyxy: [x_min, y_min, x_max, y_max] (standard CV format)
        - yxyx: [y_min, x_min, y_max, x_max] (Gemma4 box_2d format)
        - xywh: [x, y, width, height] (YOLO format)
        - cxcywh: [center_x, center_y, width, height] (COCO format)
    """
    if coord_format == "xyxy":
        return [x1, y1, x2, y2]
    elif coord_format == "yxyx":
        return [y1, x1, y2, x2]
    elif coord_format == "xywh":
        return [x1, y1, x2 - x1, y2 - y1]
    elif coord_format == "cxcywh":
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return [cx, cy, x2 - x1, y2 - y1]
    else:
        raise ValueError(f"Unknown coord_format: {coord_format}. Must be xyxy, yxyx, xywh, or cxcywh.")


class GenStrategy(str, Enum):
    """Data generation strategy."""
    ALL_IN_ONE = "all_in_one"
    PER_CLASS = "per_class"
    BOTH = "both"


class SplitMethod(str, Enum):
    """Dataset split method."""
    RANDOM = "random"
    SEQUENTIAL = "sequential"
    STRATIFIED = "stratified"


class OutputSchema(str, Enum):
    """Output record schema."""
    OPENAI_MESSAGES = "openai_messages"
    SHAREGPT = "sharegpt"


class PromptLang(str, Enum):
    """Prompt language."""
    EN = "en"
    ZH = "zh"


class PromptStyle(str, Enum):
    """Prompt style."""
    SIMPLE = "simple"
    DESCRIPTIVE = "descriptive"
    COT = "cot"


@dataclass
class DetectionFormatSpec:
    """Specification for a detection format."""
    name: str
    coordinate_format: str       # "xyxy", "xywh", or "cxcywh"
    coordinate_scale: str        # "raw", "norm_1", "norm_100", "norm_1000"
    response_structure: str      # "json_array" or "free_text"
    confidence_included: bool


FORMAT_SPECS: Dict[OutputFormat, DetectionFormatSpec] = {
    OutputFormat.LABELME_TEXT: DetectionFormatSpec(
        name="labelme_text",
        coordinate_format="xyxy",
        coordinate_scale="norm_1",
        response_structure="free_text",
        confidence_included=False,
    ),
    OutputFormat.BOX_2D_JSON: DetectionFormatSpec(
        name="box_2d_json",
        coordinate_format="xyxy",
        coordinate_scale="norm_1",
        response_structure="json_array",
        confidence_included=True,
    ),
}

DetectionPromptBuilder = Callable[[str], str]


# ---------------------------------------------------------------------------
# Prompt Template Registry
# ---------------------------------------------------------------------------

# Built-in prompt templates keyed by (lang, style, strategy)
# {class_list} placeholder is rendered as "[cat, dog]" for all_in_one or "[cat]" for per_class
_BUILTIN_PROMPT_TEMPLATES: Dict[Tuple[str, str, str], str] = {
    # English - all_in_one
    ("en", "simple", "all_in_one"): "Detect all [{class_list}].",
    ("en", "descriptive", "all_in_one"): "Please detect all [{class_list}] in the image and return their categories and bounding boxes.",
    ("en", "cot", "all_in_one"): "Please think step by step, then detect all [{class_list}] in the image and return their bounding boxes.",
    # English - per_class
    ("en", "simple", "per_class"): "Detect all [{class_list}].",
    ("en", "descriptive", "per_class"): "Please detect all [{class_list}] in the image and return their bounding boxes.",
    ("en", "cot", "per_class"): "Please think step by step, then detect all [{class_list}] in the image and return their bounding boxes.",
    # Chinese - all_in_one
    ("zh", "simple", "all_in_one"): "检测图片中所有[{class_list}]。",
    ("zh", "descriptive", "all_in_one"): "请检测图片中所有的[{class_list}], 并返回它们的边界框坐标。",
    ("zh", "cot", "all_in_one"): "请逐步思考, 然后检测图片中所有的[{class_list}], 并返回它们的边界框坐标。",
    # Chinese - per_class
    ("zh", "simple", "per_class"): "检测图片中所有的[{class_list}]。",
    ("zh", "descriptive", "per_class"): "请检测图片中所有的[{class_list}], 并返回其边界框坐标。",
    ("zh", "cot", "per_class"): "请逐步思考, 然后检测图片中所有的[{class_list}], 并返回其边界框坐标。",
}

# Built-in response templates keyed by (lang, strategy)
_BUILTIN_RESPONSE_TEMPLATES: Dict[Tuple[str, str], str] = {
    # English - all_in_one
    ("en", "all_in_one"): "I detected the following objects:\n{bbox_items}",
    # English - per_class
    ("en", "per_class"): "I found {count} [{class_list}]:\n{bbox_items}",
    # Chinese - all_in_one
    ("zh", "all_in_one"): "检测到以下目标:\n{bbox_items}",
    # Chinese - per_class
    ("zh", "per_class"): "发现{count}个[{class_list}]:\n{bbox_items}",
}

# Coordinate format description templates keyed by (lang, coord_format)
_COORD_FORMAT_DESCRIPTIONS: Dict[Tuple[str, str], str] = {
    ("en", "xyxy"): "[x_min, y_min, x_max, y_max]",
    ("en", "yxyx"): "[y_min, x_min, y_max, x_max]",
    ("en", "xywh"): "[x, y, width, height]",
    ("en", "cxcywh"): "[center_x, center_y, width, height]",
    ("zh", "xyxy"): "[x_min, y_min, x_max, y_max]",
    ("zh", "yxyx"): "[y_min, x_min, y_max, x_max]",
    ("zh", "xywh"): "[x, y, 宽, 高]",
    ("zh", "cxcywh"): "[中心x, 中心y, 宽, 高]",
}


def load_prompt_template_yaml(path: str) -> Dict[Tuple[str, str, str], str]:
    """Load custom prompt templates from a YAML file.

    YAML format:
        all_in_one_en: "Please detect all [{class_list}] ..."
        per_class_en: "Detect all [{class_list}] ..."
        all_in_one_zh: "请检测图片中所有[{class_list}] ..."
        per_class_zh: "请检测图片中所有的 [{class_list}] ..."

    Returns a dict keyed by (lang, style, custom) where style is "custom".
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for custom prompt template loading. Install with: pip install pyyaml")

    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    templates: Dict[Tuple[str, str, str], str] = {}
    for key, value in data.items():
        if not isinstance(value, str):
            continue
        # Parse key like "all_in_one_en" or "per_class_zh"
        parts = key.rsplit("_", 1)
        if len(parts) != 2:
            continue
        strategy_part, lang_part = parts
        if strategy_part == "all_in_one":
            strategy = "all_in_one"
        elif strategy_part == "per_class":
            strategy = "per_class"
        else:
            continue
        templates[(lang_part, "custom", strategy)] = value

    return templates


def build_detection_prompt(
    lang: str,
    prompt_style: str,
    gen_strategy: str,
    class_list: List[str],
    custom_templates: Optional[Dict[Tuple[str, str, str], str]] = None,
) -> str:
    """Build a detection prompt string.

    Args:
        lang: "en" or "zh"
        prompt_style: "simple", "descriptive", "cot", or "custom"
        gen_strategy: "all_in_one" or "per_class"
        class_list: list of class names
        custom_templates: optional dict from load_prompt_template_yaml()
    """
    # Render class_list placeholder
    if gen_strategy == "all_in_one":
        class_str = ", ".join(class_list)
    else:
        class_str = class_list[0] if len(class_list) == 1 else ", ".join(class_list)

    # Check custom templates first
    if prompt_style == "custom" and custom_templates:
        key = (lang, "custom", gen_strategy)
        if key in custom_templates:
            return custom_templates[key].replace("{class_list}", class_str)

    # Fall back to built-in templates
    key = (lang, prompt_style, gen_strategy)
    if key in _BUILTIN_PROMPT_TEMPLATES:
        return _BUILTIN_PROMPT_TEMPLATES[key].replace("{class_list}", class_str)

    # Ultimate fallback
    return f"Please detect all [{class_str}] in the image and return their bounding boxes."


def build_detection_response(
    lang: str,
    gen_strategy: str,
    bboxes: List[Dict[str, Any]],
    coord_format: str = "xyxy",
    coord_norm: str = "norm_1",
    target_category: Optional[str] = None,
    output_format: str = "box_2d_json",
) -> str:
    """Build a detection response string.

    Args:
        lang: "en" or "zh"
        gen_strategy: "all_in_one" or "per_class"
        bboxes: list of bbox dicts with keys: x_min, y_min, x_max, y_max, label
        coord_format: "xyxy", "xywh", "cxcywh"
        coord_norm: "raw", "norm_1", "norm_100", "norm_1000"
        target_category: for per_class mode, the category being described
        output_format: "labelme_text" or "box_2d_json"
    """
    if output_format == "box_2d_json":
        return _build_box_2d_json_response(bboxes, coord_norm, coord_format)

    # labelme_text free-text format
    coord_strs = []
    for bbox in bboxes:
        coords = _transform_and_format_coords(bbox, coord_norm, coord_format)
        label = bbox["label"]
        if gen_strategy == "per_class":
            coord_strs.append(f"- {coords}")
        else:
            coord_strs.append(f"- {label}: {coords}")

    bbox_items = "\n".join(coord_strs)

    if gen_strategy == "per_class":
        class_list = [target_category] if target_category else []
        count = len(bboxes)
        key = (lang, "per_class")
    else:
        class_list = list(set(b["label"] for b in bboxes))
        count = len(bboxes)
        key = (lang, "all_in_one")

    template = _BUILTIN_RESPONSE_TEMPLATES.get(key, "Detected objects:\n{bbox_items}")

    class_str = ", ".join(class_list)
    return template.replace("{bbox_items}", bbox_items).replace("{count}", str(count)).replace("{class_list}", class_str)


def _transform_and_format_coords(
    bbox: Dict[str, Any],
    coord_norm: str,
    coord_format: str,
) -> str:
    """Transform coordinates to target format and norm, then format as string."""
    x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]

    if coord_format == "xyxy":
        coords = [x_min, y_min, x_max, y_max]
    elif coord_format == "yxyx":
        coords = [y_min, x_min, y_max, x_max]
    elif coord_format == "xywh":
        coords = [x_min, y_min, x_max - x_min, y_max - y_min]
    elif coord_format == "cxcywh":
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        coords = [cx, cy, x_max - x_min, y_max - y_min]
    else:
        coords = [x_min, y_min, x_max, y_max]

    if coord_norm == "raw":
        return "[" + ", ".join(str(int(round(c))) for c in coords) + "]"
    elif coord_norm == "norm_1":
        return "[" + ", ".join(f"{c:.4f}" for c in coords) + "]"
    elif coord_norm in ("norm_100", "norm_1000"):
        return "[" + ", ".join(str(int(round(c))) for c in coords) + "]"
    else:
        return "[" + ", ".join(f"{c:.4f}" for c in coords) + "]"


def _build_box_2d_json_response(
    bboxes: List[Dict[str, Any]],
    coord_norm: str = "norm_1",
    coord_format: str = "xyxy",
) -> str:
    """Build a box_2d_json format response string from bbox list.

    Each bbox dict should have: x_min, y_min, x_max, y_max, label.
    Coordinates are in normalized [0,1] range (coord_norm dependent).
    """
    detections = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]

        if coord_format == "xyxy":
            raw_coords = [x_min, y_min, x_max, y_max]
        elif coord_format == "yxyx":
            raw_coords = [y_min, x_min, y_max, x_max]
        elif coord_format == "xywh":
            raw_coords = [x_min, y_min, x_max - x_min, y_max - y_min]
        elif coord_format == "cxcywh":
            raw_coords = [(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min]
        else:
            raw_coords = [x_min, y_min, x_max, y_max]

        if coord_norm == "raw":
            scaled_coords = [int(round(c)) for c in raw_coords]
        elif coord_norm == "norm_1":
            scaled_coords = [round(c, 4) for c in raw_coords]
        elif coord_norm in ("norm_100", "norm_1000"):
            scaled_coords = [int(round(c)) for c in raw_coords]
        else:
            scaled_coords = [round(c, 4) for c in raw_coords]

        detection = {
            "box_2d": scaled_coords,
            "label": bbox["label"],
        }
        if not detections:
            detection["coord_format"] = coord_format
            detection["coord_norm"] = coord_norm

        detections.append(detection)

    return json.dumps(detections, ensure_ascii=False)


def build_box_2d_json_response(bboxes: List[Dict[str, float]]) -> str:
    """Legacy function: Build a box_2d_json format response string.

    Each bbox dict should have: x_min, y_min, x_max, y_max, label.
    Returns a JSON array string with box_2d keys and normalized coords.
    Kept for backward compatibility with inference code.
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


def build_cn_detection_prompt(query: str) -> str:
    """Build Chinese detection prompt matching training format (descriptive style).

    Simple format without coordinate normalization or JSON schema instructions.
    The model recalls output format from training data.
    """
    return f"请检测图片中的{query.strip()}, 并返回边界框坐标。"


def build_en_detection_prompt(query: str) -> str:
    """Build English detection prompt matching training format (descriptive style).

    Simple format without coordinate normalization or JSON schema instructions.
    The model recalls output format from training data.
    """
    return f"Please detect all [{query.strip()}] in the image and return their bounding boxes."


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
    coord_norm: str = "auto",
) -> List[Dict[str, Any]]:
    """Parse ground truth from box_2d_json format assistant text.

    Always interprets coordinates as [x_min, y_min, x_max, y_max] (xyxy order).

    Args:
        coord_norm: coordinate normalization mode. "auto" auto-detects from
            values (norm_1 if all <= 1, else raw pixel). "norm_1000" treats
            values in [0, 1000] range, dividing by 1000 before scaling to pixels.
            "norm_1" treats values in [0, 1]. "raw" treats values as pixel coords.

    Returns list of dicts with pixel bbox: [x1, y1, x2, y2], label, confidence.
    """
    detections: List[Dict[str, Any]] = []

    def _convert(coords: List[float]) -> Tuple[int, int, int, int]:
        # Determine effective norm: explicit overrides auto-detect
        effective_norm = coord_norm
        if effective_norm == "auto":
            effective_norm = "norm_1" if _is_normalized(coords) else "raw"

        if effective_norm == "norm_1":
            return (
                int(coords[0] * img_width),
                int(coords[1] * img_height),
                int(coords[2] * img_width),
                int(coords[3] * img_height),
            )
        elif effective_norm == "norm_1000":
            return (
                int(coords[0] / 1000 * img_width),
                int(coords[1] / 1000 * img_height),
                int(coords[2] / 1000 * img_width),
                int(coords[3] / 1000 * img_height),
            )
        else:  # raw pixel coords
            return int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

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