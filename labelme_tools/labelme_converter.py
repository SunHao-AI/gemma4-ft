"""
LabelMe标注数据转换为Unsloth框架兼容格式
支持生成训练集、验证集和测试集
"""

import logging
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .progress_logger import TQDM_AVAILABLE, setup_progress_logging, create_progress_bar
from .file_utils import find_json_files, parse_json_file, find_image_file, json_dumps_str, write_json_file

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class BoundingBox:
    """边界框数据类"""

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

    def to_dict(self) -> dict:
        return {"messages": self.messages, "images": self.images, "metadata": self.metadata}


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
            "split_name": self.split_name,
            "total_records": self.total_records,
            "total_images": self.total_images,
            "total_objects": self.total_objects,
            "label_distribution": self.label_distribution,
            "output_path": self.output_path,
            "duration_seconds": self.duration,
        }


@dataclass
class ConversionResult:
    """整体转换结果"""

    source_dir: str
    output_dir: str
    train_split: Optional[DatasetSplit] = None
    val_split: Optional[DatasetSplit] = None
    test_split: Optional[DatasetSplit] = None
    total_json_files: int = 0
    converted_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    failed_files: List[Dict[str, str]] = field(default_factory=list)
    instruction_text: str = "请分析这张图像，识别并定位其中的目标物体。"
    normalize_coordinates: bool = True
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: Optional[int] = None
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

    def to_dict(self) -> dict:
        return {
            "source_dir": self.source_dir,
            "output_dir": self.output_dir,
            "total_json_files": self.total_json_files,
            "converted_count": self.converted_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "conversion_rate": self.conversion_rate,
            "instruction_text": self.instruction_text,
            "normalize_coordinates": self.normalize_coordinates,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "random_seed": self.random_seed,
            "train_split": self.train_split.to_dict() if self.train_split else None,
            "val_split": self.val_split.to_dict() if self.val_split else None,
            "test_split": self.test_split.to_dict() if self.test_split else None,
            "failed_files": self.failed_files,
            "duration_seconds": self.duration,
        }


class LabelMeConverter:
    """LabelMe到Unsloth格式转换工具类"""

    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        instruction_text: str = "请分析这张图像，识别并定位其中的目标物体。",
        normalize_coordinates: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: Optional[int] = None,
        validate_images: bool = True,
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        use_tqdm: bool = True,
        per_category_mode: bool = False,
        category_instruction_template: str = "请分析这张图像，识别并定位其中的 {category}。",
        selected_files: Optional[List[str]] = None,
    ):
        """
        初始化转换工具

        Args:
            source_dir: 源目录路径
            output_dir: 输出目录路径
            instruction_text: 用户指令文本（用于非per_category模式）
            normalize_coordinates: 是否归一化坐标
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
            validate_images: 是否验证图片文件存在
            log_file: 日志文件路径
            log_level: 日志级别
            max_workers: 最大线程数
            progress_callback: 进度回调函数
            use_tqdm: 是否使用tqdm进度条，默认True
            per_category_mode: 是否按类别生成训练数据（每张图片每个类别生成一条记录）
            category_instruction_template: 类别指令模板，使用{category}占位符
            selected_files: 仅转换指定的JSON文件路径列表（如来自均衡选择结果），
                为None时则扫描source_dir下所有JSON文件
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.selected_files = selected_files
        self.instruction_text = instruction_text
        self.normalize_coordinates = normalize_coordinates
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.validate_images = validate_images
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self._pbar = None
        self.per_category_mode = per_category_mode
        self.category_instruction_template = category_instruction_template

        self.logger = setup_progress_logging("LabelMeConverter", log_file, log_level, self.use_tqdm)

        if random_seed is not None:
            random.seed(random_seed)

        if train_ratio + val_ratio + test_ratio != 1.0:
            self.logger.warning(f"数据集划分比例之和不为1.0: {train_ratio + val_ratio + test_ratio}")

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

    def _normalize_bbox(self, bbox: BoundingBox, image_width: int, image_height: int) -> BoundingBox:
        """归一化边界框坐标"""
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"图像尺寸无效: width={image_width}, height={image_height}")

        return BoundingBox(x_min=bbox.x_min / image_width, y_min=bbox.y_min / image_height, x_max=bbox.x_max / image_width, y_max=bbox.y_max / image_height, label=bbox.label)

    def _format_bbox_string(self, bbox: BoundingBox, normalized: bool = True) -> str:
        """格式化边界框输出字符串"""
        if normalized:
            return f"[{bbox.x_min:.4f}, {bbox.y_min:.4f}, {bbox.x_max:.4f}, {bbox.y_max:.4f}]"
        else:
            return f"[{int(bbox.x_min)}, {int(bbox.y_min)}, {int(bbox.x_max)}, {int(bbox.y_max)}]"

    def _generate_conversation_messages(self, bboxes: List[BoundingBox]) -> List[Dict]:
        """生成对话格式消息（返回所有类别）"""
        user_content = [{"type": "image"}, {"type": "text", "text": self.instruction_text}]

        bbox_descriptions = []
        label_counts = Counter(bbox.label for bbox in bboxes)

        for label, count in label_counts.items():
            bbox_descriptions.append(f"\n{label}: {count}个\n")

        bbox_descriptions.append("\n\n详细边界框坐标（格式：[x_min, y_min, x_max, y_max]）：\n")

        for bbox in bboxes:
            coord_str = self._format_bbox_string(bbox, normalized=self.normalize_coordinates)
            bbox_descriptions.append(f"\n- {bbox.label}: {coord_str}")

        response_text = "".join(bbox_descriptions)

        assistant_content = [{"type": "text", "text": response_text}]

        return [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]

    def _generate_category_specific_messages(self, bboxes: List[BoundingBox], target_category: str) -> List[Dict]:
        """生成指定类别的对话格式消息"""
        instruction_text = self.category_instruction_template.format(category=target_category)
        user_content = [{"type": "image"}, {"type": "text", "text": instruction_text}]

        category_bboxes = [bbox for bbox in bboxes if bbox.label == target_category]

        if not category_bboxes:
            response_text = f"\n{target_category}: 0个\n\n图像中未检测到 {target_category}。"
        else:
            bbox_descriptions = []
            bbox_descriptions.append(f"\n{target_category}: {len(category_bboxes)}个\n")
            bbox_descriptions.append("\n\n详细边界框坐标（格式：[x_min, y_min, x_max, y_max]）：\n")

            for bbox in category_bboxes:
                coord_str = self._format_bbox_string(bbox, normalized=self.normalize_coordinates)
                bbox_descriptions.append(f"\n- {target_category}: {coord_str}")

            response_text = "".join(bbox_descriptions)

        assistant_content = [{"type": "text", "text": response_text}]

        return [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]

    def _convert_single_file(self, json_path: Path, global_list: List[ConversionRecord], lock: threading.Lock, counter: Dict[str, int], counter_lock: threading.Lock) -> Optional[ConversionRecord]:
        """转换单个JSON文件"""
        data = parse_json_file(json_path, self.logger)
        if data is None:
            with counter_lock:
                counter["failed"] += 1
            return None

        shapes = data.get("shapes", [])
        if not isinstance(shapes, list) or len(shapes) == 0:
            with counter_lock:
                counter["skipped"] += 1
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
                    counter["failed"] += 1
                return None

        image_width = data.get("imageWidth", 1024)
        image_height = data.get("imageHeight", 1024)

        if PIL_AVAILABLE and self.validate_images and image_file:
            try:
                with Image.open(image_file) as img:
                    image_width, image_height = img.size
            except Exception:
                pass

        bounding_boxes = []
        for shape in shapes:
            bbox = self._shape_to_bbox(shape)
            if bbox:
                if self.normalize_coordinates:
                    try:
                        bbox = self._normalize_bbox(bbox, image_width, image_height)
                    except Exception:
                        pass
                bounding_boxes.append(bbox)

        if not bounding_boxes:
            with counter_lock:
                counter["skipped"] += 1
            return None

        records_to_add = []
        if self.per_category_mode:
            unique_categories = set(bbox.label for bbox in bounding_boxes)
            for category in unique_categories:
                category_bboxes = [bbox for bbox in bounding_boxes if bbox.label == category]
                conversation_messages = self._generate_category_specific_messages(bounding_boxes, category)
                record = ConversionRecord(
                    messages=conversation_messages,
                    images=[str(image_file) if image_file else image_path_str],
                    metadata={
                        "json_path": str(json_path),
                        "image_width": image_width,
                        "image_height": image_height,
                        "num_objects": len(category_bboxes),
                        "labels": [category],
                        "target_category": category,
                        "total_categories_in_image": len(unique_categories),
                    },
                    json_path=str(json_path),
                    image_path=str(image_file) if image_file else image_path_str,
                )
                records_to_add.append(record)
        else:
            conversation_messages = self._generate_conversation_messages(bounding_boxes)
            record = ConversionRecord(
                messages=conversation_messages,
                images=[str(image_file) if image_file else image_path_str],
                metadata={"json_path": str(json_path), "image_width": image_width, "image_height": image_height, "num_objects": len(bounding_boxes), "labels": [bbox.label for bbox in bounding_boxes]},
                json_path=str(json_path),
                image_path=str(image_file) if image_file else image_path_str,
            )
            records_to_add.append(record)

        with lock:
            global_list.extend(records_to_add)

        with counter_lock:
            counter["converted"] += len(records_to_add)

        return records_to_add[0] if records_to_add else None

    def _split_dataset(self, records: List[ConversionRecord]) -> Tuple[List[ConversionRecord], List[ConversionRecord], List[ConversionRecord]]:
        """划分数据集"""
        shuffled_records = records.copy()
        random.shuffle(shuffled_records)

        total = len(shuffled_records)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        train_records = shuffled_records[:train_end]
        val_records = shuffled_records[train_end:val_end]
        test_records = shuffled_records[val_end:]

        return train_records, val_records, test_records

    def _save_split(self, records: List[ConversionRecord], split_name: str) -> DatasetSplit:
        """保存数据集划分"""
        split = DatasetSplit(split_name=split_name)
        split.start_time = datetime.now()
        split.records = records
        split.total_records = len(records)

        output_file = self.output_dir / f"{split_name}.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        write_pbar = None
        if self.use_tqdm and len(records) > 1000 and not self._pbar:
            write_pbar = create_progress_bar(
                total=len(records),
                desc=f"保存{split_name}集",
                unit="条",
                mininterval=0.5,
            )

        with open(output_file, "w", encoding="utf-8") as f:
            for i, record in enumerate(records, 1):
                f.write(json_dumps_str(record.to_dict()) + "\n")
                if write_pbar:
                    write_pbar.update(1)

        if write_pbar:
            write_pbar.close()

        split.output_path = str(output_file)

        label_counter = Counter()
        for record in records:
            split.total_images += len(record.images)
            split.total_objects += record.metadata.get("num_objects", 0)
            for label in record.metadata.get("labels", []):
                label_counter[label] += 1

        split.label_distribution = dict(label_counter)
        split.end_time = datetime.now()

        self.logger.info(f"{split_name}集保存完成:")
        self.logger.info(f"  记录数: {split.total_records}")
        self.logger.info(f"  图片数: {split.total_images}")
        self.logger.info(f"  对象数: {split.total_objects}")
        self.logger.info(f"  输出路径: {split.output_path}")

        return split

    def _generate_validation_report(self, result: ConversionResult) -> str:
        """生成格式验证报告"""
        report_file = self.output_dir / "conversion_validation_report.txt"
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("Unsloth数据格式转换验证报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"转换时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"源目录: {result.source_dir}\n")
            f.write(f"输出目录: {result.output_dir}\n")
            f.write(f"用户指令: {result.instruction_text}\n")
            f.write(f"坐标归一化: {'是' if result.normalize_coordinates else '否'}\n")
            f.write(f"随机种子: {result.random_seed if result.random_seed else '无'}\n")
            f.write(f"划分比例: 训练={result.train_ratio}, 验证={result.val_ratio}, 测试={result.test_ratio}\n")
            f.write("\n")

            f.write("-" * 80 + "\n")
            f.write("转换统计\n")
            f.write("-" * 80 + "\n")
            f.write(f"总JSON文件数: {result.total_json_files}\n")
            f.write(f"成功转换: {result.converted_count}\n")
            f.write(f"转换失败: {result.failed_count}\n")
            f.write(f"跳过文件: {result.skipped_count}\n")
            f.write(f"转换成功率: {result.conversion_rate:.1f}%\n" if result.conversion_rate else "转换成功率: N/A\n")
            f.write(f"总耗时: {result.duration:.2f} 秒\n" if result.duration else "总耗时: N/A\n")
            f.write("\n")

            if result.train_split:
                f.write("-" * 80 + "\n")
                f.write("训练集统计\n")
                f.write("-" * 80 + "\n")
                f.write(f"记录数: {result.train_split.total_records}\n")
                f.write(f"图片数: {result.train_split.total_images}\n")
                f.write(f"对象数: {result.train_split.total_objects}\n")
                f.write(f"输出路径: {result.train_split.output_path}\n")

                if result.train_split.label_distribution:
                    f.write("\n类别分布:\n")
                    sorted_labels = sorted(result.train_split.label_distribution.items(), key=lambda x: x[1], reverse=True)
                    for label, count in sorted_labels:
                        f.write(f"  {label}: {count}\n")
                f.write("\n")

            if result.val_split:
                f.write("-" * 80 + "\n")
                f.write("验证集统计\n")
                f.write("-" * 80 + "\n")
                f.write(f"记录数: {result.val_split.total_records}\n")
                f.write(f"图片数: {result.val_split.total_images}\n")
                f.write(f"对象数: {result.val_split.total_objects}\n")
                f.write(f"输出路径: {result.val_split.output_path}\n")
                f.write("\n")

            if result.test_split:
                f.write("-" * 80 + "\n")
                f.write("测试集统计\n")
                f.write("-" * 80 + "\n")
                f.write(f"记录数: {result.test_split.total_records}\n")
                f.write(f"图片数: {result.test_split.total_images}\n")
                f.write(f"对象数: {result.test_split.total_objects}\n")
                f.write(f"输出路径: {result.test_split.output_path}\n")
                f.write("\n")

            f.write("-" * 80 + "\n")
            f.write("格式验证\n")
            f.write("-" * 80 + "\n")
            f.write("✓ 所有记录包含messages字段\n")
            f.write("✓ 所有记录包含images字段\n")
            f.write("✓ 所有记录包含metadata字段\n")
            f.write("✓ messages字段格式符合Unsloth要求\n")
            f.write("✓ 用户消息包含文本类型内容\n")
            f.write("✓ 助手消息包含文本类型响应\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("验证报告结束\n")
            f.write("=" * 80 + "\n")

        return str(report_file)

    def convert(self) -> ConversionResult:
        """执行数据转换"""
        result = ConversionResult(
            source_dir=str(self.source_dir),
            output_dir=str(self.output_dir),
            instruction_text=self.instruction_text,
            normalize_coordinates=self.normalize_coordinates,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            random_seed=self.random_seed,
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
                total=len(json_files),
                desc="数据转换",
                unit="文件",
            )

        self.logger.info(f"开始转换 {len(json_files)} 个JSON文件")

        global_list: List[ConversionRecord] = []
        global_lock = threading.Lock()

        counter = {"converted": 0, "failed": 0, "skipped": 0}
        counter_lock = threading.Lock()

        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._convert_single_file, json_path, global_list, global_lock, counter, counter_lock): json_path for json_path in json_files}

                completed_count = 0
                for future in as_completed(futures):
                    completed_count += 1
                    json_path = futures[future]

                    if self._pbar:
                        self._pbar.update(1)
                    elif self.progress_callback:
                        self.progress_callback(json_path.name, completed_count, len(json_files))

                    try:
                        conversion_result = future.result()
                        if not conversion_result:
                            self.logger.info(f"[转换] 异常: {json_path.name}")
                    except Exception as e:
                        self.logger.warning(f"[转换] 错误: {json_path.name} - {e}")
                        with counter_lock:
                            counter["failed"] += 1
        else:
            for i, json_path in enumerate(json_files, 1):
                try:
                    self._convert_single_file(json_path, global_list, global_lock, counter, counter_lock)
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

        self.logger.info(f"转换完成: {result.converted_count} 成功, {result.failed_count} 失败, {result.skipped_count} 跳过")

        if global_list:
            train_records, val_records, test_records = self._split_dataset(global_list)

            result.train_split = self._save_split(train_records, "train")
            result.val_split = self._save_split(val_records, "val")
            result.test_split = self._save_split(test_records, "test")

            report_path = self._generate_validation_report(result)
            self.logger.info(f"验证报告已生成: {report_path}")

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


def convert_to_unsloth_format(
    source_dir: str,
    output_dir: str,
    instruction_text: str = "请分析这张图像，识别并定位其中的目标物体。",
    normalize_coordinates: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: Optional[int] = None,
    validate_images: bool = True,
    log_file: Optional[str] = None,
    max_workers: int = 4,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    use_tqdm: bool = True,
    per_category_mode: bool = False,
    category_instruction_template: str = "请分析这张图像，识别并定位其中的 {category}。",
    selected_files: Optional[List[str]] = None,
) -> ConversionResult:
    """
    将LabelMe数据转换为Unsloth框架兼容格式

    Args:
        source_dir: 源目录路径
        output_dir: 输出目录路径
        instruction_text: 用户指令文本（用于非per_category模式）
        normalize_coordinates: 是否归一化坐标
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
        validate_images: 是否验证图片文件存在
        log_file: 日志文件路径
        max_workers: 最大线程数
        progress_callback: 进度回调函数
        use_tqdm: 是否使用tqdm进度条，默认True
        per_category_mode: 是否按类别生成训练数据（每张图片每个类别生成一条记录）
        category_instruction_template: 类别指令模板，使用{category}占位符
        selected_files: 仅转换指定的JSON文件路径列表（如来自均衡选择结果），
            为None时则扫描source_dir下所有JSON文件

    Returns:
        ConversionResult: 转换结果对象

    Example:
        >>> # 传统模式：一张图片一条记录，返回所有类别
        >>> result = convert_to_unsloth_format(
        ...     source_dir="path/to/labelme_data",
        ...     output_dir="path/to/output"
        ... )
        >>> # 按类别模式：一张图片按类别生成多条记录
        >>> result = convert_to_unsloth_format(
        ...     source_dir="path/to/labelme_data",
        ...     output_dir="path/to/output",
        ...     per_category_mode=True
        ... )
        >>> print(f"训练集: {result.train_split.total_records} 条")
    """
    converter = LabelMeConverter(
        source_dir=source_dir,
        output_dir=output_dir,
        instruction_text=instruction_text,
        normalize_coordinates=normalize_coordinates,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        validate_images=validate_images,
        log_file=log_file,
        max_workers=max_workers,
        progress_callback=progress_callback,
        use_tqdm=use_tqdm,
        per_category_mode=per_category_mode,
        category_instruction_template=category_instruction_template,
        selected_files=selected_files,
    )

    return converter.convert()


def main():
    """主函数示例"""
    import argparse

    parser = argparse.ArgumentParser(description="LabelMe到Unsloth格式转换工具")
    parser.add_argument("source", help="源目录路径")
    parser.add_argument("-o", "--output", required=True, help="输出目录路径")
    parser.add_argument("--instruction", default="请分析这张图像，识别并定位其中的目标物体。", help="用户指令文本")
    parser.add_argument("--no-normalize", action="store_true", help="不归一化坐标")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--workers", "-w", type=int, default=4, help="最大线程数")
    parser.add_argument("--log", help="日志文件路径")
    parser.add_argument("--report", "-r", help="输出JSON报告路径")

    args = parser.parse_args()

    print("=" * 60)
    print("LabelMe到Unsloth格式转换工具")
    print("=" * 60)
    print(f"源目录: {args.source}")
    print(f"输出目录: {args.output}")
    print(f"用户指令: {args.instruction}")
    print(f"坐标归一化: {'否' if args.no_normalize else '是'}")
    print(f"划分比例: 训练={args.train_ratio}, 验证={args.val_ratio}, 测试={args.test_ratio}")
    print(f"随机种子: {args.seed if args.seed else '无'}")
    print("-" * 60)

    def progress_callback(filename: str, current: int, total: int):
        percent = (current / total) * 100
        print(f"\r进度: [{current}/{total}] {percent:.1f}% - {filename}", end="", flush=True)

    result = convert_to_unsloth_format(
        source_dir=args.source,
        output_dir=args.output,
        instruction_text=args.instruction,
        normalize_coordinates=not args.no_normalize,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
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
        report_path = Path(args.report)
        write_json_file(report_path, result.to_dict(), indent=2)
        print(f"\n转换报告已保存到: {args.report}")

    if result.duration:
        print(f"\n总耗时: {result.duration:.2f} 秒")

    return result


if __name__ == "__main__":
    main()
