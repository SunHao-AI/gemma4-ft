"""
LabelMe标注数据转换为Unsloth框架兼容格式

支持特性:
- 坐标归一化: raw / norm_1 / norm_100 / norm_1000
- 坐标格式: xyxy / xywh / cxcywh
- 生成策略: all_in_one / per_class / both
- Prompt模板: 多语言(en/zh)、多风格(simple/descriptive/cot)、自定义YAML
- 数据过滤: 类别白名单/黑名单、类别重映射、形状类型过滤、bbox校验
- 输出格式: OpenAI messages / ShareGPT
- 数据划分: random / sequential / stratified
- 统计输出: dataset_info.json
"""

import base64
import json
import logging
import random
import shutil
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .detection_format import (
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
    load_prompt_template_yaml,
)
from .file_utils import (
    find_image_file,
    find_json_files,
    json_dumps_str,
    parse_json_file,
    write_json_file,
)
from .progress_logger import TQDM_AVAILABLE, create_progress_bar, setup_progress_logging

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """边界框数据类 (始终以 xyxy 格式存储，归一化后的值)"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str


@dataclass
class ConversionRecord:
    """单个转换记录"""
    messages: List[Dict]
    images: List[str]
    metadata: Dict
    json_path: str
    image_path: str
    record_id: str = ""
    gen_strategy_tag: str = "all_in_one"
    is_test_only: bool = False

    def to_dict(self, schema: str = "openai_messages") -> dict:
        if schema == "sharegpt":
            # ShareGPT format: {id, image, conversations}
            human_value = self._extract_text_from_messages("user")
            if self.is_test_only:
                conversations = [{"from": "human", "value": "<image>\n" + human_value}]
            else:
                assistant_value = self._extract_text_from_messages("assistant")
                conversations = [
                    {"from": "human", "value": "<image>\n" + human_value},
                    {"from": "gpt", "value": assistant_value},
                ]
            return {
                "id": self.record_id,
                "image": self.images[0] if self.images else "",
                "conversations": conversations,
            }
        else:
            # OpenAI messages format: {messages, images, metadata}
            if self.is_test_only:
                messages = [m for m in self.messages if m.get("role") == "user"]
            else:
                messages = self.messages
            return {"messages": messages, "images": self.images, "metadata": self.metadata}

    def _extract_text_from_messages(self, role: str) -> str:
        for msg in self.messages:
            if msg.get("role") == role:
                content = msg.get("content", [])
                if isinstance(content, list):
                    return " ".join(item.get("text", "") for item in content if item.get("type") == "text")
                elif isinstance(content, str):
                    return content
        return ""


@dataclass
class DatasetSplit:
    """数据集划分结果"""
    split_name: str
    records: List[ConversionRecord] = field(default_factory=list)
    total_records: int = 0
    total_images: int = 0
    total_objects: int = 0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    output_path: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> dict:
        return {
            "images": self.total_images,
            "annotations": self.total_objects,
            "records": self.total_records,
        }


@dataclass
class ConversionResult:
    """整体转换结果"""
    source_dir: str
    output_dir: str
    config: Dict = field(default_factory=dict)
    skipped_details: Dict[str, int] = field(default_factory=lambda: {
        "empty_annotations": 0,
        "invalid_bbox": 0,
        "missing_images": 0,
    })
    total_json_files: int = 0
    converted_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    failed_files: List[Dict[str, str]] = field(default_factory=list)
    train_split: Optional[DatasetSplit] = None
    val_split: Optional[DatasetSplit] = None
    test_split: Optional[DatasetSplit] = None
    per_class_train_split: Optional[DatasetSplit] = None
    per_class_val_split: Optional[DatasetSplit] = None
    per_class_test_split: Optional[DatasetSplit] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def conversion_rate(self) -> Optional[float]:
        if self.total_json_files > 0:
            return (self.converted_count / self.total_json_files) * 100
        return None


# ---------------------------------------------------------------------------
# Split Ratio Parser
# ---------------------------------------------------------------------------

def _parse_split_ratio(split_str: str) -> Tuple[float, float, float]:
    """Parse split ratio string like "8:1:1" into three float ratios.

    Returns (train_ratio, val_ratio, test_ratio).
    """
    parts = split_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid split format: '{split_str}'. Expected format like '8:1:1'")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"Invalid split values: '{split_str}'. Values must be integers")
    total = sum(nums)
    if total == 0:
        raise ValueError("Split values sum to zero")
    return nums[0] / total, nums[1] / total, nums[2] / total


# ---------------------------------------------------------------------------
# LabelMeConverter
# ---------------------------------------------------------------------------

class LabelMeConverter:
    """LabelMe到Unsloth格式转换工具类"""

    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        # Coordinate pipeline
        coord_norm: str = "norm_1000",
        coord_format: str = "xyxy",
        # Generation strategy
        gen_strategy: str = "all_in_one",
        # Prompt system
        lang: str = "en",
        prompt_style: str = "descriptive",
        prompt_template_file: Optional[str] = None,
        # Split
        split: str = "8:1:1",
        split_method: str = "random",
        random_seed: Optional[int] = None,
        # Output
        output_schema: str = "openai_messages",
        output_format: str = "box_2d_json",
        image_path_mode: str = "relative",
        copy_images: bool = False,
        # Filtering
        class_whitelist: Optional[List[str]] = None,
        class_blacklist: Optional[List[str]] = None,
        class_remap: Optional[Dict[str, str]] = None,
        class_remap_file: Optional[str] = None,
        shape_types: Optional[List[str]] = None,
        min_bbox_size: int = 2,
        keep_empty: bool = False,
        # Test-only
        test_only: bool = False,
        # Infrastructure
        validate_images: bool = True,
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        use_tqdm: bool = True,
        selected_files: Optional[List[str]] = None,
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.selected_files = selected_files

        # Coordinate pipeline
        self.coord_norm = CoordNorm(coord_norm)
        self.coord_format = CoordFormat(coord_format)

        # Generation strategy
        self.gen_strategy = GenStrategy(gen_strategy)

        # Prompt system
        self.lang = PromptLang(lang)
        self.prompt_style = PromptStyle(prompt_style)
        self.prompt_template_file = prompt_template_file
        self.custom_templates: Dict[Tuple[str, str, str], str] = {}
        if prompt_template_file:
            self.custom_templates = load_prompt_template_yaml(prompt_template_file)
            self.prompt_style = "custom"

        # Split
        self.train_ratio, self.val_ratio, self.test_ratio = _parse_split_ratio(split)
        self.split_str = split
        self.split_method = SplitMethod(split_method)
        self.random_seed = random_seed

        # Output
        self.output_schema = OutputSchema(output_schema)
        self.output_format = OutputFormat(output_format)
        self.image_path_mode = image_path_mode
        self.copy_images = copy_images

        # Filtering
        self.class_whitelist = set(class_whitelist) if class_whitelist else None
        self.class_blacklist = set(class_blacklist) if class_blacklist else None
        self.class_remap = class_remap or {}
        if class_remap_file:
            self._load_class_remap_file(class_remap_file)
        self.shape_types = set(shape_types) if shape_types else {"rectangle", "polygon"}
        self.min_bbox_size = min_bbox_size
        self.keep_empty = keep_empty

        # Test-only
        self.test_only = test_only

        # Infrastructure
        self.validate_images = validate_images
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self._pbar = None

        self.logger = setup_progress_logging("LabelMeConverter", log_file, log_level, self.use_tqdm)

        if random_seed is not None:
            random.seed(random_seed)

        # Store config for dataset_info.json
        self._config_dict = {
            "coord_norm": coord_norm,
            "coord_format": coord_format,
            "gen_strategy": gen_strategy,
            "lang": lang,
            "prompt_style": prompt_style if not prompt_template_file else "custom",
            "prompt_template_file": prompt_template_file,
            "split": split,
            "split_method": split_method,
            "random_seed": random_seed,
            "output_schema": output_schema,
            "output_format": output_format,
            "image_path_mode": image_path_mode,
            "copy_images": copy_images,
            "class_whitelist": list(class_whitelist) if class_whitelist else None,
            "class_blacklist": list(class_blacklist) if class_blacklist else None,
            "class_remap": class_remap,
            "class_remap_file": class_remap_file,
            "shape_types": list(self.shape_types),
            "min_bbox_size": min_bbox_size,
            "keep_empty": keep_empty,
            "test_only": test_only,
            "validate_images": validate_images,
        }

    def _load_class_remap_file(self, path: str) -> None:
        """Load class remapping from a JSON file."""
        remap_path = Path(path)
        if not remap_path.exists():
            self.logger.warning(f"Class remap file not found: {path}")
            return
        try:
            with open(remap_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.class_remap.update(data)
        except (json.JSONDecodeError, OSError) as e:
            self.logger.warning(f"Failed to load class remap file: {e}")

    # -----------------------------------------------------------------------
    # Shape to BBox Conversion
    # -----------------------------------------------------------------------

    def _polygon_to_bbox(self, points: List[List[float]], label: str) -> BoundingBox:
        """多边形转换为边界框"""
        if not points or len(points) < 2:
            raise ValueError(f"多边形点数据无效: {points}")

        x_coords = [p[0] for p in points if len(p) >= 2]
        y_coords = [p[1] for p in points if len(p) >= 2]

        if not x_coords or not y_coords:
            raise ValueError(f"无法提取有效坐标: {points}")

        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)

        if x_min >= x_max or y_min >= y_max:
            raise ValueError(f"边界框坐标无效: x_min={x_min}, x_max={x_max}")

        return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, label=label)

    def _rectangle_to_bbox(self, shape: Dict) -> BoundingBox:
        """矩形转换为边界框"""
        points = shape.get("points", [])
        if len(points) < 2:
            raise ValueError(f"矩形标注点数不足: {len(points)}")

        label = shape.get("label", "")
        x_min, y_min = points[0]
        x_max, y_max = points[1]

        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, label=label)

    def _circle_to_bbox(self, shape: Dict) -> BoundingBox:
        """圆形转换为边界框"""
        points = shape.get("points", [])
        label = shape.get("label", "")

        cx, cy = points[0] if len(points) >= 1 else (0, 0)
        r = shape.get("radius", 0)

        return BoundingBox(x_min=cx - r, y_min=cy - r, x_max=cx + r, y_max=cy + r, label=label)

    def _shape_to_bbox(self, shape: Dict) -> Optional[BoundingBox]:
        """将标注形状转换为边界框"""
        label = shape.get("label", "")
        shape_type = shape.get("shape_type", "polygon")
        points = shape.get("points", [])

        try:
            if shape_type in ["polygon", "line", "point"]:
                return self._polygon_to_bbox(points, label)
            elif shape_type == "rectangle":
                return self._rectangle_to_bbox(shape)
            elif shape_type == "circle":
                return self._circle_to_bbox(shape)
            else:
                return self._polygon_to_bbox(points, label)
        except Exception as e:
            self.logger.warning(f"shape转换失败: {e}")
            return None

    # -----------------------------------------------------------------------
    # Coordinate Normalization & Transformation
    # -----------------------------------------------------------------------

    def _normalize_bbox(self, bbox: BoundingBox, image_width: int, image_height: int) -> BoundingBox:
        """归一化边界框坐标，根据 coord_norm 模式"""
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"图像尺寸无效: width={image_width}, height={image_height}")

        if self.coord_norm == CoordNorm.RAW:
            return bbox
        elif self.coord_norm == CoordNorm.NORM_1:
            return BoundingBox(
                x_min=round(bbox.x_min / image_width, 4),
                y_min=round(bbox.y_min / image_height, 4),
                x_max=round(bbox.x_max / image_width, 4),
                y_max=round(bbox.y_max / image_height, 4),
                label=bbox.label,
            )
        elif self.coord_norm == CoordNorm.NORM_100:
            return BoundingBox(
                x_min=round(bbox.x_min / image_width * 100),
                y_min=round(bbox.y_min / image_height * 100),
                x_max=round(bbox.x_max / image_width * 100),
                y_max=round(bbox.y_max / image_height * 100),
                label=bbox.label,
            )
        elif self.coord_norm == CoordNorm.NORM_1000:
            return BoundingBox(
                x_min=round(bbox.x_min / image_width * 1000),
                y_min=round(bbox.y_min / image_height * 1000),
                x_max=round(bbox.x_max / image_width * 1000),
                y_max=round(bbox.y_max / image_height * 1000),
                label=bbox.label,
            )
        else:
            raise ValueError(f"Unknown coord_norm: {self.coord_norm}")

    def _transform_coords(self, bbox: BoundingBox) -> List[float]:
        """将归一化后的 xyxy bbox 转换为目标坐标格式"""
        if self.coord_format == CoordFormat.XYXY:
            return [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]
        elif self.coord_format == CoordFormat.XYWH:
            return [bbox.x_min, bbox.y_min, bbox.x_max - bbox.x_min, bbox.y_max - bbox.y_min]
        elif self.coord_format == CoordFormat.CXCYWH:
            cx = (bbox.x_min + bbox.x_max) / 2
            cy = (bbox.y_min + bbox.y_max) / 2
            w = bbox.x_max - bbox.x_min
            h = bbox.y_max - bbox.y_min
            return [cx, cy, w, h]
        else:
            raise ValueError(f"Unknown coord_format: {self.coord_format}")

    def _format_coord_list(self, coords: List[float]) -> str:
        """格式化坐标列表为字符串"""
        if self.coord_norm == CoordNorm.RAW:
            return "[" + ", ".join(str(int(round(c))) for c in coords) + "]"
        elif self.coord_norm == CoordNorm.NORM_1:
            return "[" + ", ".join(f"{c:.4f}" for c in coords) + "]"
        elif self.coord_norm == CoordNorm.NORM_100:
            return "[" + ", ".join(str(int(round(c))) for c in coords) + "]"
        elif self.coord_norm == CoordNorm.NORM_1000:
            return "[" + ", ".join(str(int(round(c))) for c in coords) + "]"
        else:
            return "[" + ", ".join(f"{c:.4f}" for c in coords) + "]"

    # -----------------------------------------------------------------------
    # Bbox Validation
    # -----------------------------------------------------------------------

    def _validate_bbox(self, bbox: BoundingBox, image_width: int, image_height: int) -> bool:
        """Validate bounding box against anomaly rules.

        Returns True if bbox is valid, False otherwise.
        Checks: negative coords, out-of-bounds, zero area, min-size.
        """
        # Negative coordinates
        if bbox.x_min < 0 or bbox.y_min < 0:
            return False

        # Out of image bounds
        if bbox.x_max > image_width or bbox.y_max > image_height:
            return False

        # Zero or inverted area
        if bbox.x_max <= bbox.x_min or bbox.y_max <= bbox.y_min:
            return False

        # Minimum size threshold
        if (bbox.x_max - bbox.x_min) < self.min_bbox_size or (bbox.y_max - bbox.y_min) < self.min_bbox_size:
            return False

        return True

    # -----------------------------------------------------------------------
    # Image Path Resolution
    # -----------------------------------------------------------------------

    def _resolve_image_path(self, image_file: Optional[Path], json_path: Path, image_path_str: str) -> str:
        """Resolve image path based on image_path_mode."""
        if self.image_path_mode == "absolute":
            return str(image_file) if image_file else str(json_path.parent / image_path_str)

        elif self.image_path_mode == "relative":
            if image_file:
                try:
                    return str(image_file.relative_to(self.output_dir))
                except ValueError:
                    # If image is not under output_dir, store absolute path
                    return str(image_file)
            return image_path_str

        elif self.image_path_mode == "copy":
            if image_file and image_file.exists():
                images_dir = self.output_dir / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                dest = images_dir / image_file.name
                shutil.copy2(str(image_file), str(dest))
                return f"images/{image_file.name}"
            return image_path_str

        elif self.image_path_mode == "base64":
            if image_file and image_file.exists():
                try:
                    with open(str(image_file), "rb") as f:
                        return base64.b64encode(f.read()).decode("utf-8")
                except OSError:
                    return image_path_str
            return image_path_str

        else:
            return str(image_file) if image_file else image_path_str

    # -----------------------------------------------------------------------
    # Record Building
    # -----------------------------------------------------------------------

    def _build_record_all_in_one(
        self,
        bboxes: List[BoundingBox],
        image_file: Optional[Path],
        json_path: Path,
        image_path_str: str,
        image_width: int,
        image_height: int,
        is_test_only: bool = False,
    ) -> ConversionRecord:
        """Build a single record containing all categories."""
        # Determine image path
        resolved_image_path = self._resolve_image_path(image_file, json_path, image_path_str)

        # Build prompt
        class_list = list(set(b.label for b in bboxes))
        prompt_text = build_detection_prompt(
            lang=self.lang,
            prompt_style=self.prompt_style,
            gen_strategy="all_in_one",
            class_list=class_list,
            custom_templates=self.custom_templates if self.prompt_style == "custom" else None,
        )

        # Build response (only if not test_only)
        if not is_test_only:
            bbox_dicts = [{"x_min": b.x_min, "y_min": b.y_min, "x_max": b.x_max, "y_max": b.y_max, "label": b.label} for b in bboxes]
            response_text = build_detection_response(
                lang=self.lang,
                gen_strategy="all_in_one",
                bboxes=bbox_dicts,
                coord_format=self.coord_format,
                coord_norm=self.coord_norm,
                output_format=self.output_format,
            )
        else:
            response_text = ""

        # Build messages
        user_content = [{"type": "text", "text": prompt_text}]
        assistant_content = [{"type": "text", "text": response_text}] if not is_test_only else []

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ] if not is_test_only else [
            {"role": "user", "content": user_content},
        ]

        # Record ID
        stem = json_path.stem if json_path else "unknown"
        record_id = stem

        images_list = [resolved_image_path] if self.image_path_mode != "base64" else [resolved_image_path]

        metadata = {
            "json_path": str(json_path),
            "image_width": image_width,
            "image_height": image_height,
            "num_objects": len(bboxes),
            "labels": class_list,
            "output_format": self.output_format,
            "coord_norm": self.coord_norm,
            "coord_format": self.coord_format,
        }

        return ConversionRecord(
            messages=messages,
            images=images_list,
            metadata=metadata,
            json_path=str(json_path),
            image_path=str(image_file) if image_file else image_path_str,
            record_id=record_id,
            gen_strategy_tag="all_in_one",
            is_test_only=is_test_only,
        )

    def _build_record_per_class(
        self,
        bboxes: List[BoundingBox],
        target_category: str,
        image_file: Optional[Path],
        json_path: Path,
        image_path_str: str,
        image_width: int,
        image_height: int,
        is_test_only: bool = False,
    ) -> ConversionRecord:
        """Build a record for a single category."""
        resolved_image_path = self._resolve_image_path(image_file, json_path, image_path_str)

        category_bboxes = [b for b in bboxes if b.label == target_category]

        # Build prompt
        prompt_text = build_detection_prompt(
            lang=self.lang,
            prompt_style=self.prompt_style,
            gen_strategy="per_class",
            class_list=[target_category],
            custom_templates=self.custom_templates if self.prompt_style == "custom" else None,
        )

        # Build response
        if not is_test_only:
            bbox_dicts = [{"x_min": b.x_min, "y_min": b.y_min, "x_max": b.x_max, "y_max": b.y_max, "label": b.label} for b in category_bboxes]
            response_text = build_detection_response(
                lang=self.lang,
                gen_strategy="per_class",
                bboxes=bbox_dicts,
                coord_format=self.coord_format,
                coord_norm=self.coord_norm,
                target_category=target_category,
                output_format=self.output_format,
            )
        else:
            response_text = ""

        user_content = [{"type": "text", "text": prompt_text}]
        assistant_content = [{"type": "text", "text": response_text}] if not is_test_only else []

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ] if not is_test_only else [
            {"role": "user", "content": user_content},
        ]

        stem = json_path.stem if json_path else "unknown"
        record_id = f"{stem}_{target_category}"

        metadata = {
            "json_path": str(json_path),
            "image_width": image_width,
            "image_height": image_height,
            "num_objects": len(category_bboxes),
            "labels": [target_category],
            "target_category": target_category,
            "total_categories_in_image": len(set(b.label for b in bboxes)),
            "output_format": self.output_format,
            "coord_norm": self.coord_norm,
            "coord_format": self.coord_format,
        }

        return ConversionRecord(
            messages=messages,
            images=[resolved_image_path],
            metadata=metadata,
            json_path=str(json_path),
            image_path=str(image_file) if image_file else image_path_str,
            record_id=record_id,
            gen_strategy_tag="per_class",
            is_test_only=is_test_only,
        )

    # -----------------------------------------------------------------------
    # Single File Conversion
    # -----------------------------------------------------------------------

    def _convert_single_file(
        self,
        json_path: Path,
        global_list_all: List[ConversionRecord],
        global_list_per_class: List[ConversionRecord],
        lock: threading.Lock,
        counter: Dict[str, int],
        counter_lock: threading.Lock,
    ) -> Optional[List[ConversionRecord]]:
        """转换单个JSON文件"""
        data = parse_json_file(json_path, self.logger)
        if data is None:
            with counter_lock:
                counter["failed"] += 1
            return None

        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            with counter_lock:
                counter["failed"] += 1
            return None

        image_path_str = data.get("imagePath", "")
        if not image_path_str:
            with counter_lock:
                counter["skipped"] += 1
            return None

        image_file = None
        if self.validate_images:
            image_file = find_image_file(json_path, image_path_str)
            if image_file is None:
                self.logger.warning(f"图片不存在: {json_path}")
                with counter_lock:
                    counter["missing_images"] += 1
                    counter["skipped"] += 1
                return None

        image_width = data.get("imageWidth", 1024)
        image_height = data.get("imageHeight", 1024)

        if PIL_AVAILABLE and self.validate_images and image_file:
            try:
                with Image.open(image_file) as img:
                    image_width, image_height = img.size
            except Exception:
                pass

        # --- Filter pipeline ---
        # Step 1: Filter by shape type
        filtered_shapes = [s for s in shapes if s.get("shape_type", "polygon") in self.shape_types]

        # Step 2: Apply class remapping
        if self.class_remap:
            for s in filtered_shapes:
                original_label = s.get("label", "")
                if original_label in self.class_remap:
                    s["label"] = self.class_remap[original_label]

        # Step 3: Filter by class whitelist/blacklist
        if self.class_whitelist is not None:
            filtered_shapes = [s for s in filtered_shapes if s.get("label", "") in self.class_whitelist]
        if self.class_blacklist is not None:
            filtered_shapes = [s for s in filtered_shapes if s.get("label", "") not in self.class_blacklist]

        # Step 4: Convert shapes to bboxes and validate
        bounding_boxes = []
        for shape in filtered_shapes:
            bbox = self._shape_to_bbox(shape)
            if bbox:
                # Validate bbox before normalization (using raw pixel coords)
                if self.coord_norm == CoordNorm.RAW:
                    if not self._validate_bbox(bbox, image_width, image_height):
                        with counter_lock:
                            counter["invalid_bbox"] += 1
                        continue
                else:
                    # For normalized modes, validate against raw pixel coords first
                    if not self._validate_bbox(bbox, image_width, image_height):
                        with counter_lock:
                            counter["invalid_bbox"] += 1
                        continue
                # Normalize
                try:
                    bbox = self._normalize_bbox(bbox, image_width, image_height)
                except Exception as e:
                    self.logger.warning(f"归一化失败: {e}")
                    with counter_lock:
                        counter["invalid_bbox"] += 1
                    continue
                bounding_boxes.append(bbox)

        # Step 5: Handle empty annotations
        if not bounding_boxes:
            if self.keep_empty:
                # Generate empty record for negative sample
                resolved_image_path = self._resolve_image_path(image_file, json_path, image_path_str)
                prompt_text = build_detection_prompt(
                    lang=self.lang, prompt_style=self.prompt_style,
                    gen_strategy="all_in_one", class_list=["object"],
                    custom_templates=self.custom_templates if self.prompt_style == "custom" else None,
                )
                response_text = "[]" if self.output_format == OutputFormat.BOX_2D_JSON else "No objects detected."
                empty_record = ConversionRecord(
                    messages=[
                        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
                        {"role": "assistant", "content": [{"type": "text", "text": response_text}]},
                    ],
                    images=[resolved_image_path],
                    metadata={
                        "json_path": str(json_path),
                        "image_width": image_width,
                        "image_height": image_height,
                        "num_objects": 0,
                        "labels": [],
                        "output_format": self.output_format,
                        "coord_norm": self.coord_norm,
                        "coord_format": self.coord_format,
                        "is_empty_sample": True,
                    },
                    json_path=str(json_path),
                    image_path=str(image_file) if image_file else image_path_str,
                    record_id=json_path.stem + "_empty",
                    gen_strategy_tag="all_in_one",
                    is_test_only=False,
                )
                with lock:
                    global_list_all.append(empty_record)
                with counter_lock:
                    counter["converted"] += 1
                return [empty_record]
            else:
                with counter_lock:
                    counter["skipped_empty"] += 1
                    counter["skipped"] += 1
                return None

        # --- Build records ---
        records: List[ConversionRecord] = []

        if self.gen_strategy in (GenStrategy.ALL_IN_ONE, GenStrategy.BOTH):
            is_test = self.test_only
            record = self._build_record_all_in_one(
                bboxes=bounding_boxes,
                image_file=image_file,
                json_path=json_path,
                image_path_str=image_path_str,
                image_width=image_width,
                image_height=image_height,
                is_test_only=is_test,
            )
            records.append(record)
            with lock:
                global_list_all.append(record)

        if self.gen_strategy in (GenStrategy.PER_CLASS, GenStrategy.BOTH):
            unique_categories = set(b.label for b in bounding_boxes)
            for category in unique_categories:
                is_test = self.test_only
                record = self._build_record_per_class(
                    bboxes=bounding_boxes,
                    target_category=category,
                    image_file=image_file,
                    json_path=json_path,
                    image_path_str=image_path_str,
                    image_width=image_width,
                    image_height=image_height,
                    is_test_only=is_test,
                )
                records.append(record)
                with lock:
                    global_list_per_class.append(record)

        with counter_lock:
            counter["converted"] += len(records)

        return records

    # -----------------------------------------------------------------------
    # Dataset Splitting
    # -----------------------------------------------------------------------

    @staticmethod
    def _get_record_image_key(record: ConversionRecord) -> str:
        """Return a stable key for image-level grouping."""
        if record.image_path:
            return record.image_path
        if record.images:
            return record.images[0]
        if record.json_path:
            return record.json_path
        return json.dumps(record.to_dict())

    def _split_dataset(
        self,
        records: List[ConversionRecord],
    ) -> Tuple[List[ConversionRecord], List[ConversionRecord], List[ConversionRecord]]:
        """Split records into train/val/test based on split_method."""
        # Group by image to prevent leakage
        image_groups: Dict[str, List[ConversionRecord]] = {}
        image_primary_labels: Dict[str, str] = {}

        for record in records:
            image_key = self._get_record_image_key(record)
            image_groups.setdefault(image_key, []).append(record)
            # Track primary label for stratified split
            labels = record.metadata.get("labels", [])
            if labels and image_key not in image_primary_labels:
                image_primary_labels[image_key] = labels[0]

        group_keys = list(image_groups.keys())

        if self.split_method == SplitMethod.SEQUENTIAL:
            # Sort by filename, then sequential cut
            group_keys.sort()
            grouped = [image_groups[k] for k in group_keys]
        elif self.split_method == SplitMethod.STRATIFIED:
            # Stratified split: group by primary label, apply ratios within each group
            return self._split_stratified(image_groups, image_primary_labels, group_keys)
        else:
            # Random split (current default behavior)
            random.shuffle(group_keys)
            grouped = [image_groups[k] for k in group_keys]

        total_images = len(grouped)
        train_end = int(total_images * self.train_ratio)
        val_end = train_end + int(total_images * self.val_ratio)

        if self.test_ratio == 0:
            # No test split (e.g., 9:1:0)
            train_groups = grouped[:train_end]
            val_groups = grouped[train_end:]
            test_groups = []
        else:
            train_groups = grouped[:train_end]
            val_groups = grouped[train_end:val_end]
            test_groups = grouped[val_end:]

        train_records = [r for g in train_groups for r in g]
        val_records = [r for g in val_groups for r in g]
        test_records = [r for g in test_groups for r in g]

        self.logger.info(
            "按图片级划分完成: 总图片=%d, 训练=%d, 验证=%d, 测试=%d",
            total_images, len(train_groups), len(val_groups), len(test_groups),
        )

        return train_records, val_records, test_records

    def _split_stratified(
        self,
        image_groups: Dict[str, List[ConversionRecord]],
        image_primary_labels: Dict[str, str],
        group_keys: List[str],
    ) -> Tuple[List[ConversionRecord], List[ConversionRecord], List[ConversionRecord]]:
        """Stratified split ensuring proportional class distribution."""
        train_records: List[ConversionRecord] = []
        val_records: List[ConversionRecord] = []
        test_records: List[ConversionRecord] = []

        # Group images by primary label
        label_groups: Dict[str, List[str]] = {}
        for key in group_keys:
            label = image_primary_labels.get(key, "unknown")
            label_groups.setdefault(label, []).append(key)

        for label, keys in label_groups.items():
            random.shuffle(keys)
            group_list = [image_groups[k] for k in keys]
            total = len(group_list)
            train_end = int(total * self.train_ratio)
            val_end = train_end + int(total * self.val_ratio)

            if self.test_ratio == 0:
                train_records.extend(r for g in group_list[:train_end] for r in g)
                val_records.extend(r for g in group_list[train_end:] for r in g)
            else:
                train_records.extend(r for g in group_list[:train_end] for r in g)
                val_records.extend(r for g in group_list[train_end:val_end] for r in g)
                test_records.extend(r for g in group_list[val_end:] for r in g)

        return train_records, val_records, test_records

    # -----------------------------------------------------------------------
    # Save & Report
    # -----------------------------------------------------------------------

    def _save_split(
        self,
        records: List[ConversionRecord],
        split_name: str,
        output_dir: Optional[Path] = None,
        is_test: bool = False,
    ) -> DatasetSplit:
        """Save a dataset split as JSONL file."""
        out_dir = output_dir or self.output_dir
        split = DatasetSplit(split_name=split_name)
        split.start_time = datetime.now()
        split.records = records
        split.total_records = len(records)

        output_name = "valid" if split_name == "val" else split_name
        output_file = out_dir / f"{output_name}.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Mark test records as test_only if needed
        final_records = records
        if is_test and not self.test_only:
            # Test split gets test-only format only for ShareGPT schema
            if self.output_schema == OutputSchema.SHAREGPT:
                for rec in final_records:
                    rec.is_test_only = True

        write_pbar = None
        if self.use_tqdm and len(records) > 1000 and not self._pbar:
            write_pbar = create_progress_bar(
                total=len(records), desc=f"保存{split_name}集", unit="条", mininterval=0.5,
            )

        with open(output_file, "w", encoding="utf-8") as f:
            for i, record in enumerate(final_records, 1):
                f.write(json_dumps_str(record.to_dict(schema=self.output_schema)) + "\n")
                if write_pbar:
                    write_pbar.update(1)

        if write_pbar:
            write_pbar.close()

        split.output_path = str(output_file)

        label_counter = Counter()
        unique_images = set()
        for record in records:
            if record.image_path:
                unique_images.add(record.image_path)
            for image_path in record.images:
                unique_images.add(image_path)
            split.total_objects += record.metadata.get("num_objects", 0)
            for label in record.metadata.get("labels", []):
                label_counter[label] += 1

        split.total_images = len(unique_images)
        split.label_distribution = dict(label_counter)
        split.end_time = datetime.now()

        self.logger.info(f"{split_name}集保存完成:")
        self.logger.info(f"  记录数: {split.total_records}")
        self.logger.info(f"  图片数: {split.total_images}")
        self.logger.info(f"  对象数: {split.total_objects}")
        self.logger.info(f"  输出路径: {split.output_path}")

        return split

    def _generate_dataset_info(self, result: ConversionResult) -> str:
        """Generate structured dataset_info.json."""
        info_file = self.output_dir / "dataset_info.json"
        info_file.parent.mkdir(parents=True, exist_ok=True)

        # Compute per-split class distribution
        class_distribution: Dict[str, Dict[str, int]] = {}
        all_labels = set()

        for split_name, split_obj in [("train", result.train_split), ("val", result.val_split), ("test", result.test_split)]:
            if split_obj:
                all_labels.update(split_obj.label_distribution.keys())

        for label in all_labels:
            dist: Dict[str, int] = {"total": 0}
            for split_name, split_obj in [("train", result.train_split), ("val", result.val_split), ("test", result.test_split)]:
                count = 0
                if split_obj:
                    count = split_obj.label_distribution.get(label, 0)
                    dist["total"] += count
                dist[split_name] = count
            class_distribution[label] = dist

        total_images = sum(
            s.total_images if s else 0
            for s in [result.train_split, result.val_split, result.test_split]
        )
        total_annotations = sum(
            s.total_objects if s else 0
            for s in [result.train_split, result.val_split, result.test_split]
        )

        splits_dict = {}
        for split_name, split_obj in [("train", result.train_split), ("valid", result.val_split), ("test", result.test_split)]:
            if split_obj:
                splits_dict[split_name] = split_obj.to_dict()

        info_data = {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "splits": splits_dict,
            "class_distribution": class_distribution,
            "config": self._config_dict,
            "skipped": result.skipped_details,
        }

        write_json_file(info_file, info_data, indent=2)
        self.logger.info(f"dataset_info.json已生成: {info_file}")
        return str(info_file)

    # -----------------------------------------------------------------------
    # Main Convert
    # -----------------------------------------------------------------------

    def convert(self) -> ConversionResult:
        """执行数据转换"""
        result = ConversionResult(
            source_dir=str(self.source_dir),
            output_dir=str(self.output_dir),
            config=self._config_dict,
        )
        result.start_time = datetime.now()

        if self.selected_files:
            json_files = [Path(f) for f in self.selected_files]
            self.logger.info(f"使用筛选文件列表: {len(json_files)} 个文件")
        else:
            json_files = find_json_files(self.source_dir, logger=self.logger)
        result.total_json_files = len(json_files)

        if not json_files:
            self.logger.warning("未找到任何JSON文件")
            result.end_time = datetime.now()
            return result

        if self.use_tqdm and not self.progress_callback:
            self._pbar = create_progress_bar(
                total=len(json_files), desc="数据转换", unit="文件",
            )

        self.logger.info(f"开始转换 {len(json_files)} 个JSON文件")

        global_list_all: List[ConversionRecord] = []
        global_list_per_class: List[ConversionRecord] = []
        global_lock = threading.Lock()

        counter = {
            "converted": 0, "failed": 0, "skipped": 0,
            "invalid_bbox": 0, "skipped_empty": 0, "missing_images": 0,
        }
        counter_lock = threading.Lock()

        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._convert_single_file, jp, global_list_all,
                        global_list_per_class, global_lock, counter, counter_lock,
                    ): jp for jp in json_files
                }
                completed_count = 0
                for future in as_completed(futures):
                    completed_count += 1
                    json_path = futures[future]
                    if self._pbar:
                        self._pbar.update(1)
                    elif self.progress_callback:
                        self.progress_callback(json_path.name, completed_count, len(json_files))
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.warning(f"[转换] 错误: {json_path.name} - {e}")
                        with counter_lock:
                            counter["failed"] += 1
        else:
            for i, json_path in enumerate(json_files, 1):
                try:
                    self._convert_single_file(
                        json_path, global_list_all, global_list_per_class,
                        global_lock, counter, counter_lock,
                    )
                except Exception as e:
                    self.logger.warning(f"[转换] 错误: {json_path.name} - {e}")
                    with counter_lock:
                        counter["failed"] += 1
                if self._pbar:
                    self._pbar.update(1)
                elif self.progress_callback:
                    self.progress_callback(json_path.name, i, len(json_files))

        if self._pbar:
            self._pbar.close()
            self._pbar = None

        result.converted_count = counter["converted"]
        result.failed_count = counter["failed"]
        result.skipped_count = counter["skipped"]
        result.skipped_details = {
            "empty_annotations": counter["skipped_empty"],
            "invalid_bbox": counter["invalid_bbox"],
            "missing_images": counter["missing_images"],
        }

        self.logger.info(
            f"转换完成: {result.converted_count} 成功, {result.failed_count} 失败, {result.skipped_count} 跳过"
        )

        # --- Split and save ---
        def _process_records(records: List[ConversionRecord], subdir: Optional[str] = None) -> None:
            if not records:
                return
            out_dir = self.output_dir / subdir if subdir else self.output_dir
            train, val, test = self._split_dataset(records)

            result.train_split = self._save_split(train, "train", out_dir)
            result.val_split = self._save_split(val, "val", out_dir, is_test=True)
            if self.test_ratio > 0:
                result.test_split = self._save_split(test, "test", out_dir, is_test=True)

        if self.gen_strategy == GenStrategy.ALL_IN_ONE:
            _process_records(global_list_all)

        elif self.gen_strategy == GenStrategy.PER_CLASS:
            _process_records(global_list_per_class)

        elif self.gen_strategy == GenStrategy.BOTH:
            # Process all_in_one records into subdirectory
            all_in_one_result = ConversionResult(
                source_dir=str(self.source_dir),
                output_dir=str(self.output_dir / "all_in_one"),
                config=self._config_dict,
            )
            if global_list_all:
                train, val, test = self._split_dataset(global_list_all)
                result.train_split = self._save_split(train, "train", self.output_dir / "all_in_one")
                result.val_split = self._save_split(val, "val", self.output_dir / "all_in_one", is_test=True)
                if self.test_ratio > 0:
                    result.test_split = self._save_split(test, "test", self.output_dir / "all_in_one", is_test=True)

            # Process per_class records into subdirectory
            if global_list_per_class:
                train, val, test = self._split_dataset(global_list_per_class)
                result.per_class_train_split = self._save_split(train, "train", self.output_dir / "per_class")
                result.per_class_val_split = self._save_split(val, "val", self.output_dir / "per_class", is_test=True)
                if self.test_ratio > 0:
                    result.per_class_test_split = self._save_split(test, "test", self.output_dir / "per_class", is_test=True)

            # Generate dataset_info.json for each subdirectory
            self._generate_dataset_info_for_subdir(result, "all_in_one")
            self._generate_dataset_info_for_subdir(result, "per_class")

        # Generate dataset_info.json (for non-both modes)
        if self.gen_strategy != GenStrategy.BOTH:
            self._generate_dataset_info(result)

        result.end_time = datetime.now()

        self.logger.info("=" * 50)
        self.logger.info("数据转换流程完成！")
        self.logger.info(f"总JSON文件数: {result.total_json_files}")
        self.logger.info(f"成功转换: {result.converted_count}")
        self.logger.info(f"转换失败: {result.failed_count}")
        self.logger.info(f"跳过文件: {result.skipped_count}")
        if result.conversion_rate:
            self.logger.info(f"转换成功率: {result.conversion_rate:.1f}%")
        if result.duration:
            self.logger.info(f"总耗时: {result.duration:.2f} 秒")
        self.logger.info("=" * 50)

        return result

    def _generate_dataset_info_for_subdir(self, result: ConversionResult, subdir: str) -> str:
        """Generate dataset_info.json for a subdirectory (used with gen_strategy='both')."""
        subdir_path = self.output_dir / subdir
        info_file = subdir_path / "dataset_info.json"
        info_file.parent.mkdir(parents=True, exist_ok=True)

        if subdir == "all_in_one":
            train_split = result.train_split
            val_split = result.val_split
            test_split = result.test_split
        else:
            train_split = result.per_class_train_split
            val_split = result.per_class_val_split
            test_split = result.per_class_test_split

        # Compute class distribution
        class_distribution: Dict[str, Dict[str, int]] = {}
        all_labels = set()
        for split_obj in [train_split, val_split, test_split]:
            if split_obj:
                all_labels.update(split_obj.label_distribution.keys())

        for label in all_labels:
            dist: Dict[str, int] = {"total": 0}
            for split_name, split_obj in [("train", train_split), ("val", val_split), ("test", test_split)]:
                count = 0
                if split_obj:
                    count = split_obj.label_distribution.get(label, 0)
                    dist["total"] += count
                dist[split_name] = count
            class_distribution[label] = dist

        total_images = sum(s.total_images if s else 0 for s in [train_split, val_split, test_split])
        total_annotations = sum(s.total_objects if s else 0 for s in [train_split, val_split, test_split])

        splits_dict = {}
        for split_name, split_obj in [("train", train_split), ("valid", val_split), ("test", test_split)]:
            if split_obj:
                splits_dict[split_name] = split_obj.to_dict()

        config_with_strategy = dict(self._config_dict)
        config_with_strategy["gen_strategy"] = subdir

        info_data = {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "splits": splits_dict,
            "class_distribution": class_distribution,
            "config": config_with_strategy,
            "skipped": result.skipped_details,
        }

        write_json_file(info_file, info_data, indent=2)
        self.logger.info(f"dataset_info.json已生成: {info_file}")
        return str(info_file)


# ---------------------------------------------------------------------------
# Convenience Function
# ---------------------------------------------------------------------------

def convert_to_unsloth_format(
    source_dir: str,
    output_dir: str,
    coord_norm: str = "norm_1000",
    coord_format: str = "xyxy",
    gen_strategy: str = "all_in_one",
    lang: str = "en",
    prompt_style: str = "descriptive",
    prompt_template_file: Optional[str] = None,
    split: str = "8:1:1",
    split_method: str = "random",
    random_seed: Optional[int] = None,
    output_schema: str = "openai_messages",
    output_format: str = "box_2d_json",
    image_path_mode: str = "relative",
    copy_images: bool = False,
    class_whitelist: Optional[List[str]] = None,
    class_blacklist: Optional[List[str]] = None,
    class_remap: Optional[Dict[str, str]] = None,
    class_remap_file: Optional[str] = None,
    shape_types: Optional[List[str]] = None,
    min_bbox_size: int = 2,
    keep_empty: bool = False,
    test_only: bool = False,
    validate_images: bool = True,
    log_file: Optional[str] = None,
    max_workers: int = 4,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    use_tqdm: bool = True,
    selected_files: Optional[List[str]] = None,
) -> ConversionResult:
    """将LabelMe数据转换为Unsloth框架兼容格式

    Args:
        source_dir: 源目录路径
        output_dir: 输出目录路径
        coord_norm: 坐标归一化模式 ("raw"/"norm_1"/"norm_100"/"norm_1000")
        coord_format: 坐标格式 ("xyxy"/"xywh"/"cxcywh")
        gen_strategy: 生成策略 ("all_in_one"/"per_class"/"both")
        lang: Prompt语言 ("en"/"zh")
        prompt_style: Prompt风格 ("simple"/"descriptive"/"cot")
        prompt_template_file: 自定义Prompt模板YAML文件路径
        split: 数据集划分比例 (如 "8:1:1")
        split_method: 划分方法 ("random"/"sequential"/"stratified")
        random_seed: 随机种子
        output_schema: 输出格式 ("openai_messages"/"sharegpt")
        output_format: 响应内容格式 ("labelme_text"/"box_2d_json")
        image_path_mode: 图片路径模式 ("relative"/"absolute"/"base64"/"copy")
        copy_images: 是否复制图片到输出目录
        class_whitelist: 类别白名单
        class_blacklist: 类别黑名单
        class_remap: 类别重映射字典
        class_remap_file: 类别重映射JSON文件路径
        shape_types: 要处理的形状类型列表
        min_bbox_size: 最小bbox尺寸阈值
        keep_empty: 是否保留空标注图片
        test_only: 是否生成仅推理格式(无answer)
        validate_images: 是否验证图片文件存在
        log_file: 日志文件路径
        max_workers: 最大线程数
        progress_callback: 进度回调函数
        use_tqdm: 是否使用tqdm进度条
        selected_files: 仅转换指定的JSON文件路径列表

    Returns:
        ConversionResult: 转换结果对象
    """
    converter = LabelMeConverter(
        source_dir=source_dir,
        output_dir=output_dir,
        coord_norm=coord_norm,
        coord_format=coord_format,
        gen_strategy=gen_strategy,
        lang=lang,
        prompt_style=prompt_style,
        prompt_template_file=prompt_template_file,
        split=split,
        split_method=split_method,
        random_seed=random_seed,
        output_schema=output_schema,
        output_format=output_format,
        image_path_mode=image_path_mode,
        copy_images=copy_images,
        class_whitelist=class_whitelist,
        class_blacklist=class_blacklist,
        class_remap=class_remap,
        class_remap_file=class_remap_file,
        shape_types=shape_types,
        min_bbox_size=min_bbox_size,
        keep_empty=keep_empty,
        test_only=test_only,
        validate_images=validate_images,
        log_file=log_file,
        max_workers=max_workers,
        progress_callback=progress_callback,
        use_tqdm=use_tqdm,
        selected_files=selected_files,
    )
    return converter.convert()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI入口"""
    import argparse

    parser = argparse.ArgumentParser(description="LabelMe到Unsloth格式转换工具")
    parser.add_argument("source", help="源目录路径")
    parser.add_argument("-o", "--output", required=True, help="输出目录路径")

    # Coordinate pipeline
    parser.add_argument("--coord-norm", choices=["raw", "norm_1", "norm_100", "norm_1000"],
                        default="norm_1000", help="坐标归一化模式")
    parser.add_argument("--coord-format", choices=["xyxy", "xywh", "cxcywh"],
                        default="xyxy", help="坐标格式")

    # Generation strategy
    parser.add_argument("--gen-strategy", choices=["all_in_one", "per_class", "both"],
                        default="all_in_one", help="生成策略")

    # Prompt system
    parser.add_argument("--lang", choices=["en", "zh"], default="en", help="Prompt语言")
    parser.add_argument("--prompt-style", choices=["simple", "descriptive", "cot"],
                        default="descriptive", help="Prompt风格")
    parser.add_argument("--prompt-template", help="自定义Prompt模板YAML文件路径")

    # Split
    parser.add_argument("--split", default="8:1:1", help="数据集划分比例 (如 8:1:1)")
    parser.add_argument("--split-method", choices=["random", "sequential", "stratified"],
                        default="random", help="划分方法")
    parser.add_argument("--seed", type=int, help="随机种子")

    # Output
    parser.add_argument("--output-schema", choices=["openai_messages", "sharegpt"],
                        default="openai_messages", help="输出格式")
    parser.add_argument("--output-format", choices=["labelme_text", "box_2d_json"],
                        default="box_2d_json", help="响应内容格式")
    parser.add_argument("--image-path-mode", choices=["relative", "absolute", "base64", "copy"],
                        default="relative", help="图片路径模式")
    parser.add_argument("--copy-images", action="store_true", help="复制图片到输出目录")

    # Filtering
    parser.add_argument("--class-whitelist", help="类别白名单(逗号分隔)")
    parser.add_argument("--class-blacklist", help="类别黑名单(逗号分隔)")
    parser.add_argument("--class-remap", help="类别重映射JSON文件路径")
    parser.add_argument("--shape-types", default="rectangle,polygon", help="形状类型(逗号分隔)")
    parser.add_argument("--min-bbox-size", type=int, default=2, help="最小bbox尺寸阈值")
    parser.add_argument("--keep-empty", action="store_true", help="保留空标注图片")

    # Test-only
    parser.add_argument("--test-only", action="store_true", help="生成仅推理格式(无answer)")

    # Infrastructure
    parser.add_argument("--workers", "-w", type=int, default=4, help="最大线程数")
    parser.add_argument("--log", help="日志文件路径")
    parser.add_argument("--report", "-r", help="输出JSON报告路径")

    args = parser.parse_args()

    # Parse comma-separated lists
    whitelist = args.class_whitelist.split(",") if args.class_whitelist else None
    blacklist = args.class_blacklist.split(",") if args.class_blacklist else None
    shape_types_list = args.shape_types.split(",")

    print("=" * 60)
    print("LabelMe到Unsloth格式转换工具")
    print("=" * 60)
    print(f"源目录: {args.source}")
    print(f"输出目录: {args.output}")
    print(f"坐标归一化: {args.coord_norm}")
    print(f"坐标格式: {args.coord_format}")
    print(f"生成策略: {args.gen_strategy}")
    print(f"Prompt语言: {args.lang}, 风格: {args.prompt_style}")
    print(f"划分比例: {args.split}, 方法: {args.split_method}")
    print(f"输出格式: {args.output_schema}")
    print("-" * 60)

    def progress_callback(filename: str, current: int, total: int):
        percent = (current / total) * 100
        print(f"\r进度: [{current}/{total}] {percent:.1f}% - {filename}", end="", flush=True)

    result = convert_to_unsloth_format(
        source_dir=args.source,
        output_dir=args.output,
        coord_norm=args.coord_norm,
        coord_format=args.coord_format,
        gen_strategy=args.gen_strategy,
        lang=args.lang,
        prompt_style=args.prompt_style,
        prompt_template_file=args.prompt_template,
        split=args.split,
        split_method=args.split_method,
        random_seed=args.seed,
        output_schema=args.output_schema,
        output_format=args.output_format,
        image_path_mode=args.image_path_mode,
        copy_images=args.copy_images,
        class_whitelist=whitelist,
        class_blacklist=blacklist,
        class_remap_file=args.class_remap,
        shape_types=shape_types_list,
        min_bbox_size=args.min_bbox_size,
        keep_empty=args.keep_empty,
        test_only=args.test_only,
        log_file=args.log,
        max_workers=args.workers,
        progress_callback=progress_callback,
    )

    print("\n")
    print("=" * 60)
    print("转换结果汇总")
    print("=" * 60)
    print(f"总JSON文件数: {result.total_json_files}")
    print(f"成功转换: {result.converted_count}")
    print(f"转换失败: {result.failed_count}")
    print(f"跳过文件: {result.skipped_count}")
    if result.conversion_rate:
        print(f"转换成功率: {result.conversion_rate:.1f}%")

    if result.train_split:
        print(f"\n训练集: {result.train_split.total_records} 条记录")
        print(f"  输出: {result.train_split.output_path}")
    if result.val_split:
        print(f"验证集: {result.val_split.total_records} 条记录")
        print(f"  输出: {result.val_split.output_path}")
    if result.test_split:
        print(f"测试集: {result.test_split.total_records} 条记录")
        print(f"  输出: {result.test_split.output_path}")

    if args.report:
        report_data = {
            "config": result.config,
            "total_json_files": result.total_json_files,
            "converted_count": result.converted_count,
            "failed_count": result.failed_count,
            "skipped_count": result.skipped_count,
            "skipped_details": result.skipped_details,
        }
        write_json_file(Path(args.report), report_data, indent=2)
        print(f"\n转换报告已保存到: {args.report}")

    if result.duration:
        print(f"\n总耗时: {result.duration:.2f} 秒")

    return result


if __name__ == "__main__":
    main()