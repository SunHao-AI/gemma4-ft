"""
LabelMe标注数据清洗与筛选工具
验证JSON标注文件的完整性，筛选合规文件并生成清洗报告
"""

import logging
import shutil
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Set, Any
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from .progress_logger import TQDM_AVAILABLE, setup_progress_logging, create_progress_bar, PhaseProgressManager
from .file_utils import (
    find_json_files,
    parse_json_file,
    find_image_file,
    get_relative_path,
    json_loads,
    json_dumps_str,
    write_json_file,
    ORJSON_AVAILABLE,
    create_file_link,
    SUPPORTED_IMAGE_EXTENSIONS,
)


class ValidationStatus(Enum):
    """验证状态枚举"""

    VALID = "valid"
    INVALID_JSON = "invalid_json"
    MISSING_SHAPES = "missing_shapes"
    EMPTY_SHAPES = "empty_shapes"
    MISSING_IMAGE = "missing_image"
    INVALID_IMAGE_PATH = "invalid_image_path"
    DUPLICATE_ANNOTATION = "duplicate_annotation"


@dataclass
class ValidationResult:
    """单个文件的验证结果"""

    json_path: str
    status: ValidationStatus
    image_path: Optional[str] = None
    error_message: Optional[str] = None
    shapes_count: int = 0

    def is_valid(self) -> bool:
        return self.status == ValidationStatus.VALID


@dataclass
class CleaningResult:
    """清洗结果数据类"""

    total_files: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    duplicate_count: int = 0
    valid_files: List[str] = field(default_factory=list)
    invalid_files: List[Dict[str, str]] = field(default_factory=list)
    duplicate_files: List[Dict[str, str]] = field(default_factory=list)
    copied_json_files: List[str] = field(default_factory=list)
    copied_image_files: List[str] = field(default_factory=list)
    output_dir: Optional[str] = None
    report_path: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    format_conversion_result: Optional[Dict[str, Any]] = None
    integrity_check_result: Optional[Dict[str, Any]] = None
    statistics_report_result: Optional[Dict[str, Any]] = None
    phase_summary: Optional[Dict[str, Any]] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def valid_ratio(self) -> Optional[float]:
        if self.total_files > 0:
            return (self.valid_count / self.total_files) * 100
        return None

    @property
    def duplicate_ratio(self) -> Optional[float]:
        if self.total_files > 0:
            return (self.duplicate_count / self.total_files) * 100
        return None

    def to_dict(self) -> dict:
        return {
            "total_files": self.total_files,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "duplicate_count": self.duplicate_count,
            "valid_ratio": self.valid_ratio,
            "duplicate_ratio": self.duplicate_ratio,
            "valid_files": self.valid_files,
            "invalid_files": self.invalid_files,
            "duplicate_files": self.duplicate_files,
            "copied_json_files": self.copied_json_files,
            "copied_image_files": self.copied_image_files,
            "output_dir": self.output_dir,
            "report_path": self.report_path,
            "duration_seconds": self.duration,
            "format_conversion_result": self.format_conversion_result,
            "integrity_check_result": self.integrity_check_result,
            "statistics_report_result": self.statistics_report_result,
            "phase_summary": self.phase_summary,
        }


class LabelMeCleaner:
    """LabelMe标注数据清洗工具类"""

    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        preserve_structure: bool = True,
        copy_images: bool = True,
        deduplicate: bool = True,
        generate_report: bool = True,
        report_path: Optional[str] = None,
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        use_tqdm: bool = True,
        max_workers: int = 4,
        enable_format_conversion: bool = True,
        enable_integrity_check: bool = True,
        enable_statistics_report: bool = True,
        format_output_dir: Optional[str] = None,
        format_subdir_name: str = "adv_label",
        cleaned_subdir_name: str = "cleaned_data",
        pretty_format_json: bool = False,
        label_mapping: Optional[Dict[str, str]] = None,
        download_remote_images: bool = False,
        remote_image_download_dir: Optional[str] = None,
    ):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.preserve_structure = preserve_structure
        self.copy_images = copy_images
        self.deduplicate = deduplicate
        self.generate_report = generate_report
        self.report_path = Path(report_path) if report_path else None
        self.progress_callback = progress_callback
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self.max_workers = max_workers
        self.enable_format_conversion = enable_format_conversion
        self.enable_integrity_check = enable_integrity_check
        self.enable_statistics_report = enable_statistics_report
        self.format_subdir_name = format_subdir_name
        self.cleaned_subdir_name = cleaned_subdir_name

        if format_output_dir:
            self.format_output_dir = Path(format_output_dir)
        else:
            self.format_output_dir = self.target_dir / self.format_subdir_name

        self.cleaned_data_dir = self.target_dir / self.cleaned_subdir_name

        self.pretty_format_json = pretty_format_json
        self.label_mapping = label_mapping or {}
        self.download_remote_images = download_remote_images
        self.remote_image_download_dir = Path(remote_image_download_dir) if remote_image_download_dir else self.target_dir / "downloaded_images"
        self._pbar = None
        self._phase_manager = None

        self.logger = setup_progress_logging("LabelMeCleaner", log_file, log_level, self.use_tqdm)

    def _validate_json_structure(self, json_path: Path) -> ValidationResult:
        """
        验证JSON文件结构

        Args:
            json_path: JSON文件路径

        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(json_path=str(json_path), status=ValidationStatus.VALID)

        data = parse_json_file(json_path, self.logger)
        if data is None:
            result.status = ValidationStatus.INVALID_JSON
            result.error_message = "JSON文件解析失败"
            return result

        if "shapes" not in data:
            result.status = ValidationStatus.MISSING_SHAPES
            result.error_message = "缺少'shapes'字段"
            return result

        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            result.status = ValidationStatus.MISSING_SHAPES
            result.error_message = "'shapes'字段不是数组类型"
            return result

        if len(shapes) == 0:
            result.status = ValidationStatus.EMPTY_SHAPES
            result.error_message = "'shapes'数组为空，无标注数据"
            return result

        result.shapes_count = len(shapes)

        image_path_str = data.get("imagePath", "")
        if not image_path_str:
            result.status = ValidationStatus.INVALID_IMAGE_PATH
            result.error_message = "缺少'imagePath'字段"
            return result

        result.image_path = image_path_str

        return result

    def _validate_file(self, json_path: Path) -> ValidationResult:
        """
        完整验证单个JSON文件

        Args:
            json_path: JSON文件路径

        Returns:
            ValidationResult: 验证结果
        """
        result = self._validate_json_structure(json_path)

        if not result.is_valid():
            return result

        image_file = find_image_file(json_path, result.image_path, strict_name_match=True)

        if image_file is None:
            result.status = ValidationStatus.MISSING_IMAGE
            result.error_message = f"图片文件不存在: {result.image_path}"
            return result

        result.image_path = str(image_file)
        return result

    def _copy_valid_file(self, json_path: Path, image_path: Optional[Path]) -> tuple:
        """
        复制合规文件到目标目录

        图片文件始终复制到JSON文件所在的类别文件夹中,
        确保类别文件夹结构统一性

        所有清洗后的标注数据统一存放在 cleaned_data_dir 子目录中

        Args:
            json_path: JSON文件路径
            image_path: 图片文件路径

        Returns:
            tuple: (json目标路径, 图片目标路径)
        """
        if self.preserve_structure:
            relative_json = get_relative_path(json_path, self.source_dir)
            target_json = self.cleaned_data_dir / relative_json
        else:
            target_json = self.cleaned_data_dir / json_path.name

        target_json.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(json_path, target_json)
            copied_json = str(target_json)
        except Exception as e:
            self.logger.error(f"复制JSON文件失败: {json_path} - {e}")
            copied_json = None

        copied_image = None
        if self.copy_images and image_path:
            target_image = target_json.parent / image_path.name

            target_image.parent.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy2(image_path, target_image)
                copied_image = str(target_image)
            except Exception as e:
                self.logger.error(f"复制图片文件失败: {image_path} - {e}")

        return copied_json, copied_image

    def _generate_report(self, result: CleaningResult) -> str:
        """
        生成清洗报告

        Args:
            result: 清洗结果

        Returns:
            str: 报告文件路径
        """
        if self.report_path:
            report_file = self.report_path
        else:
            report_file = self.target_dir / f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("LabelMe标注数据清洗报告\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"清洗时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"源目录: {self.source_dir}\n")
            f.write(f"目标目录: {self.target_dir}\n")
            f.write(f"去重模式: {'开启' if self.deduplicate else '关闭'}\n")
            f.write(f"耗时: {result.duration:.2f} 秒\n" if result.duration else "耗时: N/A\n")
            f.write("\n")

            f.write("-" * 70 + "\n")
            f.write("统计摘要\n")
            f.write("-" * 70 + "\n")
            f.write(f"总文件数: {result.total_files}\n")
            f.write(f"合规文件: {result.valid_count} ({result.valid_ratio:.1f}%)\n" if result.valid_ratio else f"合规文件: {result.valid_count}\n")
            f.write(f"不合规文件: {result.invalid_count}\n")
            f.write(f"重复标注文件: {result.duplicate_count} ({result.duplicate_ratio:.1f}%)\n" if result.duplicate_ratio else f"重复标注文件: {result.duplicate_count}\n")
            f.write(f"复制JSON文件数: {len(result.copied_json_files)}\n")
            f.write(f"复制图片文件数: {len(result.copied_image_files)}\n")
            f.write("\n")

            if result.duplicate_files:
                f.write("-" * 70 + "\n")
                f.write("重复标注文件详情\n")
                f.write("-" * 70 + "\n")

                image_groups: Dict[str, List[Dict]] = {}
                for item in result.duplicate_files:
                    image_file = item.get("image_file", "unknown")
                    if image_file not in image_groups:
                        image_groups[image_file] = []
                    image_groups[image_file].append(item)

                for image_file, items in image_groups.items():
                    f.write(f"\n图片: {Path(image_file).name}\n")
                    f.write(f"  重复标注数: {len(items)}\n")
                    for item in items:
                        f.write(f"    - {Path(item['file']).name}\n")
                        f.write(f"      原因: {item['reason']}\n")

                f.write("\n")

            if result.invalid_files:
                f.write("-" * 70 + "\n")
                f.write("不合规文件详情\n")
                f.write("-" * 70 + "\n")

                status_groups: Dict[str, List[Dict]] = {}
                for item in result.invalid_files:
                    status = item.get("status", "unknown")
                    if status not in status_groups:
                        status_groups[status] = []
                    status_groups[status].append(item)

                for status, items in status_groups.items():
                    f.write(f"\n[{status}] ({len(items)} 个文件)\n")
                    for item in items:
                        f.write(f"  文件: {item['file']}\n")
                        f.write(f"  原因: {item['reason']}\n")

                f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("报告结束\n")
            f.write("=" * 70 + "\n")

        return str(report_file)

    def _download_remote_image(self, image_url: str, target_dir: Path, json_path: Path) -> Optional[Path]:
        """
        下载远程图片到本地目录

        Args:
            image_url: 图片URL地址
            target_dir: 目标保存目录
            json_path: JSON文件路径（用于生成文件名）

        Returns:
            Optional[Path]: 下载后的本地路径，失败返回None
        """
        import hashlib
        import urllib.request
        import urllib.error

        try:
            url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
            url_ext = Path(image_url).suffix or ".jpg"
            safe_ext = url_ext.split("?")[0][:5] if "?" in url_ext else url_ext[:5]

            target_filename = f"{json_path.stem}_{url_hash}{safe_ext}"
            target_path = target_dir / target_filename

            if target_path.exists():
                self.logger.info(f"图片已存在: {target_path}")
                return target_path

            target_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"下载图片: {image_url[:50]}...")

            request = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

            with urllib.request.urlopen(request, timeout=30) as response:
                image_data = response.read()
                if len(image_data) < 100:
                    raise ValueError("下载的图片数据太小，可能损坏")

                with open(target_path, "wb") as f:
                    f.write(image_data)

                self.logger.info(f"下载成功: {target_path}")
                return target_path

        except urllib.error.URLError as e:
            self.logger.warning(f"网络错误，无法下载图片: {image_url[:30]} - {e}")
            return None
        except urllib.error.HTTPError as e:
            self.logger.warning(f"HTTP错误，无法下载图片: {image_url[:30]} - {e}")
            return None
        except Exception as e:
            self.logger.warning(f"下载图片失败: {image_url[:30]} - {e}")
            return None

    def _extract_image_path(self, data: Dict, json_path: Path) -> Optional[str]:
        """
        从JSON数据中提取图片路径（支持imageUrl和imagePath两种格式）

        Args:
            data: JSON解析后的数据
            json_path: JSON文件路径

        Returns:
            Optional[str]: 图片路径（本地路径或URL），None表示无法获取
        """
        image_url = data.get("imageUrl", "")
        image_path_str = data.get("imagePath", "")

        if image_url and isinstance(image_url, str):
            if self.download_remote_images:
                downloaded_path = self._download_remote_image(image_url, self.remote_image_download_dir, json_path)
                if downloaded_path:
                    return str(downloaded_path)
            data["_image_url"] = image_url
            data["_image_source"] = "remote"
            return image_url

        if image_path_str and isinstance(image_path_str, str):
            data["_image_source"] = "local"
            return image_path_str

        return None

    def _apply_label_mapping(self, shapes: List[Dict]) -> List[Dict]:
        """
        应用类别映射到shapes列表

        Args:
            shapes: 标注shapes列表

        Returns:
            List[Dict]: 映射后的shapes列表
        """
        if not self.label_mapping:
            return shapes

        mapped_shapes = []
        mapping_applied_count = 0

        for shape in shapes:
            if not isinstance(shape, dict):
                continue

            original_label = shape.get("label", "")
            if original_label and original_label in self.label_mapping:
                mapped_label = self.label_mapping[original_label]
                shape_copy = shape.copy()
                shape_copy["label"] = mapped_label
                shape_copy["_original_label"] = original_label
                shape_copy["_label_mapped"] = True
                mapped_shapes.append(shape_copy)
                mapping_applied_count += 1
            else:
                mapped_shapes.append(shape)

        if mapping_applied_count > 0:
            self.logger.debug(f"应用类别映射: {mapping_applied_count} 个标注")

        return mapped_shapes

    def _is_name_matched(self, json_path: Path, image_path: Path) -> bool:
        """
        判断JSON文件名是否与图片文件名匹配

        Args:
            json_path: JSON文件路径
            image_path: 图片文件路径

        Returns:
            bool: 是否匹配
        """
        json_stem = json_path.stem.lower()
        image_stem = image_path.stem.lower()
        return json_stem == image_stem

    def _select_primary_annotation(self, annotations: List[ValidationResult]) -> ValidationResult:
        """
        从多个指向同一图片的标注中选择主要标注文件

        选择规则：
        1. 优先选择JSON文件名与图片文件名相同的标注
        2. 若无同名标注，选择shapes_count最多的标注
        3. 若shapes_count相同，选择文件名较短或字典序靠前的

        Args:
            annotations: 指向同一图片的标注验证结果列表

        Returns:
            ValidationResult: 主要标注的验证结果
        """
        if len(annotations) == 1:
            return annotations[0]

        matched_annotations = []
        for ann in annotations:
            json_path = Path(ann.json_path)
            image_path = Path(ann.image_path)
            if self._is_name_matched(json_path, image_path):
                matched_annotations.append(ann)

        if matched_annotations:
            return matched_annotations[0]

        sorted_annotations = sorted(annotations, key=lambda x: (-x.shapes_count, len(Path(x.json_path).stem), Path(x.json_path).stem.lower()))

        return sorted_annotations[0]

    def _convert_format(self, valid_files: List[str], result: CleaningResult) -> Dict[str, Any]:
        """
        数据格式转换阶段（多线程并行）

        将清洗后的 LabelMe JSON 文件转换为标准化格式，
        添加额外元数据字段并规范化标注数据结构，
        支持类别映射和imageUrl处理

        文件保存策略：
        - 当 preserve_structure=True 时，保留源文件的相对路径结构
        - 处理 a/b/c.json 文件时，转换后保存至 format_output_dir/a/b/c.json
        - 对应的图片文件以符号链接形式保存至 format_output_dir/a/b/c.jpg

        Args:
            valid_files: 有效文件列表
            result: 清洗结果

        Returns:
            Dict: 格式转换结果统计
        """
        conversion_result = {
            "total_files": len(valid_files),
            "converted_files": 0,
            "failed_files": 0,
            "failed_details": [],
            "output_dir": str(self.format_output_dir) if self.format_output_dir else str(self.target_dir),
            "label_mapping_applied": len(self.label_mapping) > 0,
            "label_mapping_count": 0,
            "image_url_detected": 0,
            "remote_images_downloaded": 0,
            "image_links_created": 0,
            "image_links_failed": 0,
            "preserve_structure": self.preserve_structure,
        }

        if not valid_files:
            conversion_result["message"] = "无有效文件需要转换"
            return conversion_result

        output_dir = self.format_output_dir or self.target_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_cleaned_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        json_indent = 2 if self.pretty_format_json else None

        converted_files = 0
        failed_files = 0
        failed_details: List[Dict] = []
        label_mapping_count = 0
        image_url_detected = 0
        remote_images_downloaded = 0
        image_links_created = 0
        image_links_failed = 0
        converted_lock = threading.Lock()
        failed_lock = threading.Lock()
        details_lock = threading.Lock()
        mapping_lock = threading.Lock()
        url_lock = threading.Lock()
        download_lock = threading.Lock()
        image_link_lock = threading.Lock()

        def _convert_single_file(json_file: str):
            nonlocal converted_files, failed_files, failed_details, label_mapping_count, image_url_detected, remote_images_downloaded, image_links_created, image_links_failed
            json_path = Path(json_file)
            file_converted = 0
            file_failed = 0
            file_error = None
            file_mapping = 0
            file_url = 0
            file_download = 0
            file_link_created = 0
            file_link_failed = 0

            try:
                data = parse_json_file(json_path)
                if data is None:
                    file_failed = 1
                    file_error = "JSON解析失败"
                    self.logger.warning(f"格式转换失败: {json_path.name} - JSON解析失败")
                    return

                data["_cleaned_at"] = batch_cleaned_at
                data["_source_path"] = str(json_path)
                if "_cleaning_status" not in data:
                    data["_cleaning_status"] = "validated"

                shapes = data.get("shapes", [])

                if self.label_mapping:
                    mapped_shapes = self._apply_label_mapping(shapes)
                    file_mapping = sum(1 for s in mapped_shapes if s.get("_label_mapped"))
                    data["shapes"] = mapped_shapes
                    data["_label_mapping_applied"] = True
                    data["_label_mapping_count"] = file_mapping

                for shape in data.get("shapes", []):
                    if "label" in shape and "_label_normalized" not in shape:
                        shape["_label_normalized"] = shape["label"].lower().strip()
                    if "shape_type" not in shape:
                        shape_type = shape.get("type", "rectangle")
                        shape["shape_type"] = shape_type

                image_path_str = self._extract_image_path(data, json_path)
                source_image_path = None
                if image_path_str:
                    if data.get("_image_source") == "remote":
                        file_url = 1
                        if self.download_remote_images and Path(image_path_str).exists():
                            file_download = 1
                            data["_downloaded_image_path"] = image_path_str
                            source_image_path = Path(image_path_str)
                    else:
                        source_image_path = find_image_file(json_path, image_path_str)

                if self.preserve_structure:
                    relative_json = get_relative_path(json_path, self.source_dir)
                    target_json = output_dir / relative_json
                else:
                    target_json = output_dir / json_path.name

                write_json_file(target_json, data, indent=json_indent)
                file_converted = 1

                if self.copy_images and source_image_path and source_image_path.exists():
                    if self.preserve_structure:
                        relative_image = get_relative_path(source_image_path, self.source_dir)
                        target_image = output_dir / relative_image
                    else:
                        target_image = target_json.parent / source_image_path.name

                    success, method = create_file_link(
                        source_image_path,
                        target_image,
                        link_type="auto",
                        logger=self.logger,
                    )
                    if success:
                        file_link_created = 1
                        data["_image_link_path"] = str(target_image)
                        data["_image_link_method"] = method
                    else:
                        file_link_failed = 1
                        self.logger.warning(f"图片链接创建失败: {source_image_path} -> {target_image}")

            except Exception as e:
                file_failed = 1
                file_error = str(e)
                self.logger.warning(f"格式转换失败: {json_path.name} - {e}")

            with converted_lock:
                converted_files += file_converted
            with failed_lock:
                failed_files += file_failed
            with mapping_lock:
                label_mapping_count += file_mapping
            with url_lock:
                image_url_detected += file_url
            with download_lock:
                remote_images_downloaded += file_download
            with image_link_lock:
                image_links_created += file_link_created
                image_links_failed += file_link_failed
            if file_error:
                with details_lock:
                    failed_details.append(
                        {
                            "file": str(json_path),
                            "error": file_error,
                        }
                    )

            if self._phase_manager:
                self._phase_manager.update(1)

        if self.max_workers > 1 and len(valid_files) > 1:
            self.logger.info(f"使用 {self.max_workers} 个线程并行格式转换")
            if self.label_mapping:
                self.logger.info(f"应用类别映射: {len(self.label_mapping)} 个映射规则")
            if self.download_remote_images:
                self.logger.info(f"启用远程图片下载功能")
            if self.preserve_structure:
                self.logger.info(f"保留源文件相对路径结构")
            if self.copy_images:
                self.logger.info(f"图片文件将创建符号链接(跨平台兼容)")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                executor.map(_convert_single_file, valid_files)
        else:
            for json_file in valid_files:
                _convert_single_file(json_file)

        conversion_result["converted_files"] = converted_files
        conversion_result["failed_files"] = failed_files
        conversion_result["failed_details"] = failed_details
        conversion_result["label_mapping_count"] = label_mapping_count
        conversion_result["image_url_detected"] = image_url_detected
        conversion_result["remote_images_downloaded"] = remote_images_downloaded
        conversion_result["image_links_created"] = image_links_created
        conversion_result["image_links_failed"] = image_links_failed
        conversion_result["success_rate"] = conversion_result["converted_files"] / conversion_result["total_files"] * 100 if conversion_result["total_files"] > 0 else 0

        return conversion_result

    def _verify_integrity(self, valid_files: List[str], result: CleaningResult) -> Dict[str, Any]:
        """
        数据完整性验证阶段（多线程并行）

        对清洗后的数据进行二次验证，确保：
        1. JSON 文件结构完整性
        2. 图片文件存在且可访问
        3. 标注数据与图片尺寸匹配

        Args:
            valid_files: 有效文件列表
            result: 清洗结果

        Returns:
            Dict: 完整性验证结果统计
        """
        integrity_result = {
            "total_files": len(valid_files),
            "passed_files": 0,
            "failed_files": 0,
            "warnings": 0,
            "issues": [],
        }

        if not valid_files:
            integrity_result["message"] = "无有效文件需要验证"
            return integrity_result

        passed_files = 0
        failed_files = 0
        warnings = 0
        issues: List[Dict] = []
        passed_lock = threading.Lock()
        failed_lock = threading.Lock()
        warning_lock = threading.Lock()
        issues_lock = threading.Lock()

        def _verify_single_file(json_file: str):
            nonlocal passed_files, failed_files, warnings, issues
            json_path = Path(json_file)
            file_issues = []
            passed = True
            has_warning = False
            file_passed = 0
            file_failed = 0
            file_warning = 0

            data = parse_json_file(json_path)
            if data is None:
                file_failed = 1
                with issues_lock:
                    issues.append({"file": str(json_path), "passed": False, "issues": ["JSON解析失败"]})
                with failed_lock:
                    failed_files += 1
                if self._phase_manager:
                    self._phase_manager.update(1)
                return

            required_fields = ["shapes", "imagePath"]
            for field in required_fields:
                if field not in data:
                    file_issues.append(f"缺少必填字段: {field}")
                    passed = False

            shapes = data.get("shapes", [])
            if not shapes:
                file_issues.append("shapes 数组为空")
                has_warning = True

            for i, shape in enumerate(shapes):
                if "label" not in shape:
                    file_issues.append(f"shape[{i}] 缺少 label 字段")
                    has_warning = True
                shape_type = shape.get("shape_type", shape.get("type", ""))
                if shape_type in ["rectangle", "polygon", "line", "point", "circle"]:
                    points = shape.get("points", [])
                    if not points:
                        file_issues.append(f"shape[{i}] 缺少 points 数据")
                        passed = False

            image_path_str = data.get("imagePath", "")
            if image_path_str:
                image_file = find_image_file(json_path, image_path_str, strict_name_match=False)
                if image_file is None:
                    file_issues.append(f"图片文件不存在: {image_path_str}")
                    has_warning = True
                else:
                    try:
                        with open(image_file, "rb") as img_f:
                            img_data = img_f.read()
                            if len(img_data) < 100:
                                file_issues.append("图片文件可能损坏（文件过小）")
                                has_warning = True
                    except Exception as img_e:
                        file_issues.append(f"图片文件读取失败: {img_e}")
                        has_warning = True

            if passed and not has_warning:
                file_passed = 1
            elif passed and has_warning:
                file_passed = 1
                file_warning = 1
            else:
                file_failed = 1

            if file_issues:
                with issues_lock:
                    issues.append({"file": str(json_path), "passed": passed, "issues": file_issues})

            with passed_lock:
                passed_files += file_passed
            with failed_lock:
                failed_files += file_failed
            with warning_lock:
                warnings += file_warning

            if self._phase_manager:
                self._phase_manager.update(1)

        if self.max_workers > 1 and len(valid_files) > 1:
            self.logger.info(f"使用 {self.max_workers} 个线程并行验证完整性")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                executor.map(_verify_single_file, valid_files)
        else:
            for json_file in valid_files:
                _verify_single_file(json_file)

        integrity_result["passed_files"] = passed_files
        integrity_result["failed_files"] = failed_files
        integrity_result["warnings"] = warnings
        integrity_result["issues"] = issues
        integrity_result["pass_rate"] = integrity_result["passed_files"] / integrity_result["total_files"] * 100 if integrity_result["total_files"] > 0 else 0

        return integrity_result

    def _generate_statistics_report(self, valid_files: List[str], result: CleaningResult) -> Dict[str, Any]:
        """
        统计分析报告阶段（多线程并行）

        对清洗后的数据生成详细的统计分析报告：
        1. 类别分布统计
        2. 标注数量分布
        3. 文件大小统计
        4. 数据质量指标

        Args:
            valid_files: 有效文件列表
            result: 清洗结果

        Returns:
            Dict: 统计分析结果
        """
        stats_result = {
            "total_files": len(valid_files),
            "processed_files": 0,
            "failed_files": 0,
            "label_distribution": {},
            "annotation_counts": {
                "total_annotations": 0,
                "avg_per_file": 0,
                "max_per_file": 0,
                "min_per_file": 0,
            },
            "file_size_stats": {
                "total_size_bytes": 0,
                "avg_size_bytes": 0,
            },
            "shape_type_distribution": {},
        }

        if not valid_files:
            stats_result["message"] = "无有效文件需要统计"
            return stats_result

        annotation_counts_list: List[int] = []
        file_sizes: List[int] = []
        shape_types: Dict[str, int] = {}
        label_distribution: Dict[str, int] = {}
        processed_files = 0
        failed_files = 0
        total_annotations = 0
        total_size_bytes = 0

        ann_list_lock = threading.Lock()
        size_list_lock = threading.Lock()
        shape_lock = threading.Lock()
        label_lock = threading.Lock()
        processed_lock = threading.Lock()
        failed_lock = threading.Lock()
        total_ann_lock = threading.Lock()
        total_size_lock = threading.Lock()

        def _stats_single_file(json_file: str):
            nonlocal processed_files, failed_files, total_annotations, total_size_bytes
            json_path = Path(json_file)
            file_annotations = 0
            file_size = 0
            file_labels: Dict[str, int] = {}
            file_shapes: Dict[str, int] = {}
            file_processed = 0
            file_failed = 0

            try:
                file_size = json_path.stat().st_size

                data = parse_json_file(json_path)
                if data is None:
                    file_failed = 1
                    self.logger.warning(f"统计处理失败: {json_path.name} - JSON解析失败")
                    raise ValueError("JSON解析失败")

                shapes = data.get("shapes", [])
                file_annotations = len(shapes)

                for shape in shapes:
                    label = shape.get("label", "unknown")
                    file_labels[label] = file_labels.get(label, 0) + 1

                    shape_type = shape.get("shape_type", shape.get("type", "unknown"))
                    file_shapes[shape_type] = file_shapes.get(shape_type, 0) + 1

                file_processed = 1

            except Exception as e:
                file_failed = 1
                self.logger.warning(f"统计处理失败: {json_path.name} - {e}")

            with size_list_lock:
                file_sizes.append(file_size)
            with ann_list_lock:
                annotation_counts_list.append(file_annotations)
            with shape_lock:
                for st, cnt in file_shapes.items():
                    shape_types[st] = shape_types.get(st, 0) + cnt
            with label_lock:
                for lbl, cnt in file_labels.items():
                    label_distribution[lbl] = label_distribution.get(lbl, 0) + cnt
            with processed_lock:
                processed_files += file_processed
            with failed_lock:
                failed_files += file_failed
            with total_ann_lock:
                total_annotations += file_annotations
            with total_size_lock:
                total_size_bytes += file_size

            if self._phase_manager:
                self._phase_manager.update(1)

        if self.max_workers > 1 and len(valid_files) > 1:
            self.logger.info(f"使用 {self.max_workers} 个线程并行统计")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                executor.map(_stats_single_file, valid_files)
        else:
            for json_file in valid_files:
                _stats_single_file(json_file)

        stats_result["processed_files"] = processed_files
        stats_result["failed_files"] = failed_files
        stats_result["label_distribution"] = label_distribution
        stats_result["annotation_counts"]["total_annotations"] = total_annotations
        stats_result["file_size_stats"]["total_size_bytes"] = total_size_bytes

        if annotation_counts_list:
            stats_result["annotation_counts"]["avg_per_file"] = sum(annotation_counts_list) / len(annotation_counts_list)
            stats_result["annotation_counts"]["max_per_file"] = max(annotation_counts_list)
            stats_result["annotation_counts"]["min_per_file"] = min(annotation_counts_list)

        if file_sizes:
            stats_result["file_size_stats"]["avg_size_bytes"] = sum(file_sizes) / len(file_sizes)

        stats_result["shape_type_distribution"] = shape_types

        sorted_labels = sorted(stats_result["label_distribution"].items(), key=lambda x: x[1], reverse=True)
        stats_result["label_distribution_sorted"] = [{"label": k, "count": v} for k, v in sorted_labels]

        stats_result["unique_labels"] = len(stats_result["label_distribution"])

        return stats_result

    def clean(self) -> CleaningResult:
        result = CleaningResult()
        result.start_time = datetime.now()
        result.output_dir = str(self.target_dir)

        self.target_dir.mkdir(parents=True, exist_ok=True)

        json_files = find_json_files(self.source_dir, logger=self.logger)
        result.total_files = len(json_files)

        if not json_files:
            self.logger.warning("未找到任何JSON文件")
            result.end_time = datetime.now()
            return result

        phases = ["validation", "deduplication", "copy"]
        if self.enable_format_conversion:
            phases.append("format_conversion")
        if self.enable_integrity_check:
            phases.append("integrity_check")
        if self.enable_statistics_report:
            phases.append("statistics")

        self._phase_manager = PhaseProgressManager(phases, self.use_tqdm)

        self.logger.info(f"开始清洗流程，共 {len(phases)} 个阶段")

        validation_results: List[ValidationResult] = []
        self._phase_manager.start_phase("validation", len(json_files))
        self.logger.info(f"验证阶段: 处理 {len(json_files)} 个JSON文件")

        if self.max_workers > 1:
            self.logger.info(f"使用 {self.max_workers} 个线程并行验证")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._validate_file, json_path): json_path for json_path in json_files}
                for future in as_completed(futures):
                    validation_result = future.result()
                    validation_results.append(validation_result)
                    if not validation_result.is_valid():
                        self.logger.debug(f"[验证] 异常: {Path(futures[future]).name} ({validation_result.status.value})")
                    self._phase_manager.update(1)
        else:
            for i, json_path in enumerate(json_files, 1):
                validation_result = self._validate_file(json_path)
                validation_results.append(validation_result)
                if not validation_result.is_valid():
                    self.logger.debug(f"[验证] 异常: {json_path.name} ({validation_result.status.value})")
                self._phase_manager.update(1)

        self._phase_manager.complete_phase("validation")
        self.logger.info(f"验证完成: {len(validation_results)} 个文件")

        image_to_annotations: Dict[str, List[ValidationResult]] = {}
        for vr in validation_results:
            if vr.is_valid() and vr.image_path:
                image_key = str(Path(vr.image_path).resolve())
                if image_key not in image_to_annotations:
                    image_to_annotations[image_key] = []
                image_to_annotations[image_key].append(vr)

        primary_annotations: Set[str] = set()
        deduplicate_count = len(image_to_annotations)

        self._phase_manager.start_phase("deduplication", deduplicate_count)
        self.logger.info(f"去重阶段: 处理 {deduplicate_count} 个图片组")

        if self.deduplicate:
            for image_key, annotations in image_to_annotations.items():
                if len(annotations) > 1:
                    primary = self._select_primary_annotation(annotations)
                    primary_annotations.add(primary.json_path)
                    self.logger.debug(f"[去重] 图片 {Path(image_key).name} 有 {len(annotations)} 个标注，保留: {Path(primary.json_path).name}")
                    for ann in annotations:
                        if ann.json_path != primary.json_path:
                            ann.status = ValidationStatus.DUPLICATE_ANNOTATION
                            ann.error_message = f"重复标注文件，图片 {Path(image_key).name} 已有同名标注 {Path(primary.json_path).name}"
                else:
                    primary_annotations.add(annotations[0].json_path)
                self._phase_manager.update(1)
        else:
            for annotations in image_to_annotations.values():
                for ann in annotations:
                    primary_annotations.add(ann.json_path)
                self._phase_manager.update(1)

        self._phase_manager.complete_phase("deduplication")

        valid_count = sum(1 for vr in validation_results if vr.is_valid())
        self._phase_manager.start_phase("copy", valid_count)
        self.logger.info(f"复制阶段: 处理 {valid_count} 个合规文件")

        valid_vrs = [vr for vr in validation_results if vr.is_valid()]
        duplicate_vrs = [vr for vr in validation_results if vr.status == ValidationStatus.DUPLICATE_ANNOTATION]
        invalid_vrs = [vr for vr in validation_results if not vr.is_valid() and vr.status != ValidationStatus.DUPLICATE_ANNOTATION]

        result.duplicate_count = len(duplicate_vrs)
        result.duplicate_files = [{"file": vr.json_path, "status": vr.status.value, "reason": vr.error_message, "image_file": vr.image_path} for vr in duplicate_vrs]
        result.invalid_count = len(invalid_vrs)
        result.invalid_files = [{"file": vr.json_path, "status": vr.status.value, "reason": vr.error_message} for vr in invalid_vrs]

        result.valid_count = valid_count
        result.valid_files = [vr.json_path for vr in valid_vrs]

        copied_json_files: List[str] = []
        copied_image_files: List[str] = []
        json_lock = threading.Lock()
        image_lock = threading.Lock()
        copy_counter = 0
        counter_lock = threading.Lock()

        def _copy_single_valid(vr: ValidationResult):
            nonlocal copy_counter
            image_path = Path(vr.image_path) if vr.image_path else None
            copied_json, copied_image = self._copy_valid_file(Path(vr.json_path), image_path)
            if copied_json:
                with json_lock:
                    copied_json_files.append(copied_json)
            if copied_image:
                with image_lock:
                    copied_image_files.append(copied_image)
            with counter_lock:
                copy_counter += 1
                self._phase_manager.update(1)

        if self.max_workers > 1 and len(valid_vrs) > 1:
            self.logger.info(f"使用 {self.max_workers} 个线程并行复制文件")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                executor.map(_copy_single_valid, valid_vrs)
        else:
            for vr in valid_vrs:
                _copy_single_valid(vr)

        result.copied_json_files = copied_json_files
        result.copied_image_files = copied_image_files

        self._phase_manager.complete_phase("copy")

        if self.enable_format_conversion and result.valid_files:
            self._phase_manager.start_phase("format_conversion", len(result.valid_files))
            self.logger.info(f"格式转换阶段: 处理 {len(result.valid_files)} 个文件")
            conversion_result = self._convert_format(result.valid_files, result)
            result.format_conversion_result = conversion_result
            self._phase_manager.complete_phase("format_conversion")
            self.logger.info(f"格式转换完成: {conversion_result['converted_files']}/{conversion_result['total_files']}")

        if self.enable_integrity_check and result.valid_files:
            self._phase_manager.start_phase("integrity_check", len(result.valid_files))
            self.logger.info(f"完整性验证阶段: 检查 {len(result.valid_files)} 个文件")
            integrity_result = self._verify_integrity(result.copied_json_files, result)
            result.integrity_check_result = integrity_result
            self._phase_manager.complete_phase("integrity_check")
            self.logger.info(f"完整性验证完成: {integrity_result['passed_files']}/{integrity_result['total_files']} 通过")

        if self.enable_statistics_report and result.valid_files:
            self._phase_manager.start_phase("statistics", len(result.valid_files))
            self.logger.info(f"统计分析阶段: 分析 {len(result.valid_files)} 个文件")
            stats_result = self._generate_statistics_report(result.copied_json_files, result)
            result.statistics_report_result = stats_result
            self._phase_manager.complete_phase("statistics")
            self.logger.info(f"统计分析完成: {stats_result['unique_labels']} 个类别")

        result.end_time = datetime.now()

        if self.generate_report:
            report_file = self._generate_report(result)
            result.report_path = report_file
            self.logger.info(f"清洗报告已生成: {report_file}")

        phase_summary = self._phase_manager.complete_all(show_summary=True)
        result.phase_summary = phase_summary

        self.logger.info("=" * 50)
        self.logger.info("清洗完成！")
        self.logger.info(f"总文件数: {result.total_files}")
        self.logger.info(f"合规文件: {result.valid_count}")
        self.logger.info(f"不合规文件: {result.invalid_count}")
        self.logger.info(f"重复标注文件: {result.duplicate_count}")
        if result.valid_ratio:
            self.logger.info(f"合规率: {result.valid_ratio:.1f}%")
        if result.duplicate_ratio:
            self.logger.info(f"重复率: {result.duplicate_ratio:.1f}%")
        if result.duration:
            self.logger.info(f"耗时: {result.duration:.2f} 秒")
        self.logger.info("=" * 50)

        return result


def clean_labelme_data(
    source_dir: str,
    target_dir: str,
    preserve_structure: bool = True,
    copy_images: bool = True,
    deduplicate: bool = True,
    generate_report: bool = True,
    report_path: Optional[str] = None,
    log_file: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    use_tqdm: bool = True,
    max_workers: int = 4,
    enable_format_conversion: bool = True,
    enable_integrity_check: bool = True,
    enable_statistics_report: bool = True,
    format_output_dir: Optional[str] = None,
    format_subdir_name: str = "adv_label",
    cleaned_subdir_name: str = "cleaned_data",
    pretty_format_json: bool = False,
    label_mapping: Optional[Dict[str, str]] = None,
    download_remote_images: bool = False,
    remote_image_download_dir: Optional[str] = None,
) -> CleaningResult:
    """
    清洗LabelMe标注数据

    Args:
        source_dir: 源目录路径
        target_dir: 目标目录路径
        preserve_structure: 是否保留目录结构
        copy_images: 是否复制图片文件
        deduplicate: 是否去除重复标注文件
        generate_report: 是否生成清洗报告
        report_path: 报告文件路径
        log_file: 日志文件路径
        progress_callback: 进度回调函数
        use_tqdm: 是否使用tqdm进度条
        max_workers: 最大线程数
        enable_format_conversion: 是否启用格式转换
        enable_integrity_check: 是否启用完整性验证
        enable_statistics_report: 是否启用统计分析报告
        format_output_dir: 格式转换输出目录(可选,若未指定则使用target_dir/format_subdir_name)
        format_subdir_name: 格式转换子目录名称,默认"adv_label"
        cleaned_subdir_name: 清洗后标注数据子目录名称,默认"cleaned_data"
        pretty_format_json: 是否格式化JSON输出
        label_mapping: 类别映射字典 {"原始类别": "目标类别"}
        download_remote_images: 是否下载远程图片（imageUrl）
        remote_image_download_dir: 远程图片下载目录

    Returns:
        CleaningResult: 清洗结果对象
    """
    cleaner = LabelMeCleaner(
        source_dir=source_dir,
        target_dir=target_dir,
        preserve_structure=preserve_structure,
        copy_images=copy_images,
        deduplicate=deduplicate,
        generate_report=generate_report,
        report_path=report_path,
        log_file=log_file,
        progress_callback=progress_callback,
        use_tqdm=use_tqdm,
        max_workers=max_workers,
        enable_format_conversion=enable_format_conversion,
        enable_integrity_check=enable_integrity_check,
        enable_statistics_report=enable_statistics_report,
        format_output_dir=format_output_dir,
        format_subdir_name=format_subdir_name,
        cleaned_subdir_name=cleaned_subdir_name,
        pretty_format_json=pretty_format_json,
        label_mapping=label_mapping,
        download_remote_images=download_remote_images,
        remote_image_download_dir=remote_image_download_dir,
    )

    return cleaner.clean()


def main():
    """主函数示例"""
    import argparse

    parser = argparse.ArgumentParser(description="LabelMe标注数据清洗工具")
    parser.add_argument("source", help="源目录路径")
    parser.add_argument("-o", "--output", required=True, help="目标目录路径")
    parser.add_argument("--no-structure", action="store_true", help="不保留目录结构")
    parser.add_argument("--no-images", action="store_true", help="不复制图片文件")
    parser.add_argument("--no-deduplicate", action="store_true", help="不去除重复标注文件")
    parser.add_argument("--no-report", action="store_true", help="不生成清洗报告")
    parser.add_argument("--report", help="指定报告文件路径")
    parser.add_argument("--log", help="日志文件路径")

    args = parser.parse_args()

    print("=" * 60)
    print("LabelMe标注数据清洗工具")
    print("=" * 60)
    print(f"源目录: {args.source}")
    print(f"目标目录: {args.output}")
    print(f"保留目录结构: {'否' if args.no_structure else '是'}")
    print(f"复制图片文件: {'否' if args.no_images else '是'}")
    print(f"去重模式: {'关闭' if args.no_deduplicate else '开启'}")
    print("-" * 60)

    def progress_callback(filename: str, current: int, total: int):
        percent = (current / total) * 100
        print(f"\r进度: [{current}/{total}] {percent:.1f}% - {filename}", end="", flush=True)

    result = clean_labelme_data(
        source_dir=args.source,
        target_dir=args.output,
        preserve_structure=not args.no_structure,
        copy_images=not args.no_images,
        deduplicate=not args.no_deduplicate,
        generate_report=not args.no_report,
        report_path=args.report,
        log_file=args.log,
        progress_callback=progress_callback,
    )

    print("\n")
    print("=" * 60)
    print("清洗结果汇总")
    print("=" * 60)
    print(f"总文件数: {result.total_files}")
    print(f"合规文件: {result.valid_count}")
    print(f"不合规文件: {result.invalid_count}")
    print(f"重复标注文件: {result.duplicate_count}")
    if result.valid_ratio:
        print(f"合规率: {result.valid_ratio:.1f}%")
    if result.duplicate_ratio:
        print(f"重复率: {result.duplicate_ratio:.1f}%")
    print(f"复制JSON文件: {len(result.copied_json_files)}")
    print(f"复制图片文件: {len(result.copied_image_files)}")

    if result.duplicate_files:
        print("\n重复标注文件统计:")
        image_counts: Dict[str, int] = {}
        for item in result.duplicate_files:
            image_file = item.get("image_file", "unknown")
            image_counts[image_file] = image_counts.get(image_file, 0) + 1
        for image_file, count in image_counts.items():
            print(f"  图片 {Path(image_file).name}: {count} 个重复标注")

    if result.invalid_files:
        print("\n不合规文件分类统计:")
        status_counts: Dict[str, int] = {}
        for item in result.invalid_files:
            status = item.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        for status, count in status_counts.items():
            print(f"  - {status}: {count} 个")

    if result.report_path:
        print(f"\n清洗报告: {result.report_path}")

    if result.duration:
        print(f"\n总耗时: {result.duration:.2f} 秒")

    return result


@dataclass
class LabelStatistics:
    """类别统计数据类"""

    source_dir: str
    total_json_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    skipped_no_imageurl: int = 0
    skipped_parse_error: int = 0
    label_counts: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def total_labels(self) -> int:
        return len(self.label_counts)

    @property
    def total_label_instances(self) -> int:
        total = 0
        for label, count_dict in self.label_counts.items():
            for count_str, files in count_dict.items():
                count = int(count_str)
                total += count * len(files)
        return total

    def get_label_summary(self) -> Dict[str, Dict[str, int]]:
        summary = {}
        for label, count_dict in self.label_counts.items():
            summary[label] = {
                "total_files": sum(len(files) for files in count_dict.values()),
                "total_instances": sum(int(count) * len(files) for count, files in count_dict.items()),
                "max_per_file": max(int(count) for count in count_dict.keys()) if count_dict else 0,
                "min_per_file": min(int(count) for count in count_dict.keys()) if count_dict else 0,
            }
        return summary

    def to_dict(self) -> dict:
        return {
            "source_dir": self.source_dir,
            "total_json_files": self.total_json_files,
            "processed_files": self.processed_files,
            "skipped_files": self.skipped_files,
            "skipped_no_imageurl": self.skipped_no_imageurl,
            "skipped_parse_error": self.skipped_parse_error,
            "total_labels": self.total_labels,
            "total_label_instances": self.total_label_instances,
            "label_counts": {label: {count: list(files) for count, files in count_dict.items()} for label, count_dict in self.label_counts.items()},
            "label_summary": self.get_label_summary(),
            "duration_seconds": self.duration,
        }

    def to_structured_dict(self) -> dict:
        """
        生成结构化的有序字典格式

        格式规范：
        1. 顶层key与to_dict保持一致
        2. 类别名称按字母顺序（A-Z）升序排列
        3. 每个类别下的数量键按数字1,2,3...顺序排列
        4. label_summary按字母顺序排列

        Returns:
            dict: 结构化的有序字典
        """
        sorted_labels = sorted(self.label_counts.keys(), key=lambda x: x.lower())

        structured_label_counts = {}
        for label in sorted_labels:
            count_dict = self.label_counts[label]
            sorted_counts = sorted(count_dict.keys(), key=lambda x: int(x))

            structured_counts = {}
            for count in sorted_counts:
                files = sorted(count_dict[count])
                structured_counts[count] = files

            structured_label_counts[label] = structured_counts

        summary = self.get_label_summary()
        sorted_summary_keys = sorted(summary.keys(), key=lambda x: x.lower())
        structured_label_summary = {}
        for label in sorted_summary_keys:
            structured_label_summary[label] = summary[label]

        return {
            "source_dir": self.source_dir,
            "total_json_files": self.total_json_files,
            "processed_files": self.processed_files,
            "skipped_files": self.skipped_files,
            "skipped_no_imageurl": self.skipped_no_imageurl,
            "skipped_parse_error": self.skipped_parse_error,
            "total_labels": self.total_labels,
            "total_label_instances": self.total_label_instances,
            "label_counts": structured_label_counts,
            "label_summary": structured_label_summary,
            "duration_seconds": self.duration,
        }


class LabelMeLabelStatistics:
    """LabelMe标注类别统计工具类"""

    def __init__(
        self,
        source_dir: str,
        recursive: bool = True,
        use_relative_path: bool = True,
        max_workers: int = 4,
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        use_tqdm: bool = True,
    ):
        """
        初始化类别统计工具

        Args:
            source_dir: 源目录路径
            recursive: 是否递归遍历子目录，默认True
            use_relative_path: 是否使用相对路径，默认True
            max_workers: 最大线程数，默认4
            log_file: 日志文件路径，None则不记录到文件
            log_level: 日志级别
            progress_callback: 进度回调函数，参数为(文件名, 当前索引, 总数)
            use_tqdm: 是否使用tqdm进度条，默认True
        """
        self.source_dir = Path(source_dir)
        self.recursive = recursive
        self.use_relative_path = use_relative_path
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self._pbar = None

        self.logger = setup_progress_logging("LabelMeLabelStatistics", log_file, log_level, self.use_tqdm)

    def _get_file_path_str(self, file_path: Path) -> str:
        """获取文件路径字符串"""
        if self.use_relative_path:
            try:
                return str(file_path.relative_to(self.source_dir))
            except ValueError:
                return str(file_path)
        else:
            return str(file_path.resolve())

    def _check_imageurl(self, data: Dict) -> bool:
        """
        检查是否包含imageUrl字段

        Args:
            data: JSON解析后的数据

        Returns:
            bool: 是否包含imageUrl
        """
        return "imageUrl" in data and data["imageUrl"]

    def _count_labels_in_file(self, data: Dict) -> Dict[str, int]:
        """
        统计文件中各label的出现次数

        Args:
            data: JSON解析后的数据

        Returns:
            Dict[str, int]: label及其出现次数
        """
        label_counts: Dict[str, int] = {}

        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            return label_counts

        for shape in shapes:
            if not isinstance(shape, dict):
                continue

            label = shape.get("label", "")
            if not label:
                continue

            label_counts[label] = label_counts.get(label, 0) + 1

        return label_counts

    def _update_global_dict(self, global_dict: Dict[str, Dict[str, Set[str]]], label_counts: Dict[str, int], file_path_str: str, lock: threading.Lock):
        """
        更新全局统计字典（线程安全）

        Args:
            global_dict: 全局统计字典
            label_counts: 当前文件的label统计
            file_path_str: 文件路径字符串
            lock: 线程锁
        """
        with lock:
            for label, count in label_counts.items():
                if label not in global_dict:
                    global_dict[label] = {}

                count_key = str(count)
                if count_key not in global_dict[label]:
                    global_dict[label][count_key] = set()

                global_dict[label][count_key].add(file_path_str)

    def _process_single_file(self, json_path: Path, global_dict: Dict[str, Dict[str, Set[str]]], lock: threading.Lock, counter: Dict[str, int], counter_lock: threading.Lock) -> Dict:
        """
        处理单个JSON文件

        Args:
            json_path: JSON文件路径
            global_dict: 全局统计字典
            lock: 全局字典线程锁
            counter: 计数器字典
            counter_lock: 计数器线程锁

        Returns:
            Dict: 处理结果
        """
        result = {"json_path": str(json_path), "success": False, "has_imageurl": False, "label_counts": {}, "error_type": None}

        data = parse_json_file(json_path, self.logger)

        if data is None:
            result["error_type"] = "parse_error"
            with counter_lock:
                counter["skipped_parse_error"] += 1
                counter["skipped_files"] += 1
            return result

        if not self._check_imageurl(data):
            result["error_type"] = "no_imageurl"
            with counter_lock:
                counter["skipped_no_imageurl"] += 1
                counter["skipped_files"] += 1
            return result

        label_counts = self._count_labels_in_file(data)

        if not label_counts:
            result["error_type"] = "no_shapes"
            with counter_lock:
                counter["skipped_files"] += 1
            return result

        file_path_str = self._get_file_path_str(json_path)
        self._update_global_dict(global_dict, label_counts, file_path_str, lock)

        result["success"] = True
        result["has_imageurl"] = True
        result["label_counts"] = label_counts

        with counter_lock:
            counter["processed_files"] += 1

        return result

    def statistics(self) -> LabelStatistics:
        """
        执行类别统计（支持多线程）

        Returns:
            LabelStatistics: 统计结果对象
        """
        result = LabelStatistics(source_dir=str(self.source_dir))
        result.start_time = datetime.now()

        json_files = find_json_files(self.source_dir, self.recursive, self.logger)
        result.total_json_files = len(json_files)

        if not json_files:
            self.logger.warning("未找到任何JSON文件")
            result.end_time = datetime.now()
            return result

        self.logger.info(f"开始统计 {len(json_files)} 个JSON文件的类别信息")
        self.logger.info(f"使用 {self.max_workers} 个线程并行处理")

        if self.use_tqdm and not self.progress_callback:
            self._pbar = create_progress_bar(
                total=len(json_files),
                desc="类别统计",
                unit="文件",
            )

        global_dict: Dict[str, Dict[str, Set[str]]] = {}
        global_lock = threading.Lock()

        counter = {"processed_files": 0, "skipped_files": 0, "skipped_no_imageurl": 0, "skipped_parse_error": 0}
        counter_lock = threading.Lock()

        completed_count = 0
        completed_lock = threading.Lock()

        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._process_single_file, json_path, global_dict, global_lock, counter, counter_lock): json_path for json_path in json_files}

                for future in as_completed(futures):
                    json_path = futures[future]
                    with completed_lock:
                        completed_count += 1

                    if self._pbar:
                        self._pbar.update(1)
                    elif self.progress_callback:
                        with completed_lock:
                            self.progress_callback(json_path.name, completed_count, len(json_files))

                    try:
                        process_result = future.result()
                        if process_result["success"]:
                            self.logger.debug(f"[{completed_count}/{len(json_files)}] 完成: {json_path.name} - {len(process_result['label_counts'])} 个类别")
                        elif process_result["error_type"] == "no_imageurl":
                            self.logger.debug(f"[{completed_count}/{len(json_files)}] 跳过(无imageUrl): {json_path.name}")
                        elif process_result["error_type"] == "parse_error":
                            self.logger.debug(f"[{completed_count}/{len(json_files)}] 跳过(解析错误): {json_path.name}")
                    except Exception as e:
                        self.logger.warning(f"处理异常: {json_path} - {e}")
        else:
            for i, json_path in enumerate(json_files, 1):
                process_result = self._process_single_file(json_path, global_dict, global_lock, counter, counter_lock)

                if process_result["success"]:
                    self.logger.debug(f"[{i}/{len(json_files)}] 完成: {json_path.name} - " f"{len(process_result['label_counts'])} 个类别")
                elif process_result["error_type"] == "no_imageurl":
                    self.logger.debug(f"[{i}/{len(json_files)}] 跳过(无imageUrl): {json_path.name}")
                elif process_result["error_type"] == "parse_error":
                    self.logger.debug(f"[{i}/{len(json_files)}] 跳过(解析错误): {json_path.name}")

                if self._pbar:
                    self._pbar.update(1)
                elif self.progress_callback:
                    self.progress_callback(json_path.name, i, len(json_files))
                else:
                    self.logger.debug(f"[{i}/{len(json_files)}] 已处理: {json_path.name}")

        result.processed_files = counter["processed_files"]
        result.skipped_files = counter["skipped_files"]
        result.skipped_no_imageurl = counter["skipped_no_imageurl"]
        result.skipped_parse_error = counter["skipped_parse_error"]
        result.label_counts = global_dict
        result.end_time = datetime.now()

        self.logger.info("=" * 50)
        self.logger.info("统计完成！")
        self.logger.info(f"总JSON文件数: {result.total_json_files}")
        self.logger.info(f"有效处理文件: {result.processed_files}")
        self.logger.info(f"跳过文件: {result.skipped_files}")
        self.logger.info(f"  - 无imageUrl: {result.skipped_no_imageurl}")
        self.logger.info(f"  - 解析错误: {result.skipped_parse_error}")
        self.logger.info(f"类别总数: {result.total_labels}")
        self.logger.info(f"标注实例总数: {result.total_label_instances}")
        if result.duration:
            self.logger.info(f"耗时: {result.duration:.2f} 秒")
        self.logger.info("=" * 50)

        if self._pbar:
            self._pbar.close()
            self._pbar = None

        return result


def statistics_labelme_labels(
    source_dir: str,
    recursive: bool = True,
    use_relative_path: bool = True,
    max_workers: int = 4,
    log_file: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    use_tqdm: bool = True,
) -> LabelStatistics:
    """
    统计LabelMe标注文件中的类别信息

    Args:
        source_dir: 源目录路径
        recursive: 是否递归遍历子目录，默认True
        use_relative_path: 是否使用相对路径，默认True
        max_workers: 最大线程数，默认4
        log_file: 日志文件路径，None则不记录到文件
        progress_callback: 进度回调函数，参数为(文件名, 当前索引, 总数)
        use_tqdm: 是否使用tqdm进度条，默认True

    Returns:
        LabelStatistics: 统计结果对象

    Example:
        >>> result = statistics_labelme_labels("path/to/source", max_workers=8)
        >>> print(f"类别总数: {result.total_labels}")
        >>> for label, summary in result.get_label_summary().items():
        ...     print(f"{label}: {summary['total_instances']} 个实例, {summary['total_files']} 个文件")
    """
    stats = LabelMeLabelStatistics(
        source_dir=source_dir, recursive=recursive, use_relative_path=use_relative_path, max_workers=max_workers, log_file=log_file, progress_callback=progress_callback, use_tqdm=use_tqdm
    )

    return stats.statistics()


def statistics_main():
    """类别统计主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="LabelMe标注类别统计工具")
    parser.add_argument("source", help="源目录路径")
    parser.add_argument("--no-recursive", action="store_true", help="不递归遍历子目录")
    parser.add_argument("--absolute-path", action="store_true", help="使用绝对路径")
    parser.add_argument("--workers", "-w", type=int, default=4, help="最大线程数，默认4")
    parser.add_argument("--log", help="日志文件路径")
    parser.add_argument("--output", "-o", help="输出统计结果到JSON文件")

    args = parser.parse_args()

    print("=" * 60)
    print("LabelMe标注类别统计工具")
    print("=" * 60)
    print(f"源目录: {args.source}")
    print(f"递归遍历: {'否' if args.no_recursive else '是'}")
    print(f"路径格式: {'绝对路径' if args.absolute_path else '相对路径'}")
    print(f"线程数: {args.workers}")
    print("-" * 60)

    def progress_callback(filename: str, current: int, total: int):
        percent = (current / total) * 100
        print(f"\r进度: [{current}/{total}] {percent:.1f}% - {filename}", end="", flush=True)

    result = statistics_labelme_labels(
        source_dir=args.source, recursive=not args.no_recursive, use_relative_path=not args.absolute_path, max_workers=args.workers, log_file=args.log, progress_callback=progress_callback
    )

    print("\n")
    print("=" * 60)
    print("统计结果汇总")
    print("=" * 60)
    print(f"总JSON文件数: {result.total_json_files}")
    print(f"有效处理文件: {result.processed_files}")
    print(f"跳过文件: {result.skipped_files}")
    print(f"  - 无imageUrl: {result.skipped_no_imageurl}")
    print(f"  - 解析错误: {result.skipped_parse_error}")
    print(f"类别总数: {result.total_labels}")
    print(f"标注实例总数: {result.total_label_instances}")

    if result.label_counts:
        print("\n类别详情:")
        summary = result.get_label_summary()
        sorted_summary = sorted(summary.items(), key=lambda x: x[1]["total_instances"], reverse=True)
        for label, info in sorted_summary:
            print(f"  {label}:")
            print(f"    - 总实例数: {info['total_instances']}")
            print(f"    - 文件数: {info['total_files']}")
            print(f"    - 单文件最大: {info['max_per_file']}")
            print(f"    - 单文件最小: {info['min_per_file']}")

    if args.output:
        output_path = Path(args.output)
        write_json_file(output_path, result.to_structured_dict(), indent=2)
        print(f"\n统计结果已保存到: {args.output}")

    if result.duration:
        print(f"\n总耗时: {result.duration:.2f} 秒")

    return result


@dataclass
class FilterCopyResult:
    """筛选复制结果数据类"""

    statistics_file: str
    source_dir: str
    target_dir: str
    total_files_in_stats: int = 0
    copied_files: int = 0
    skipped_files: int = 0
    missing_files: List[str] = field(default_factory=list)
    copied_file_paths: List[str] = field(default_factory=list)
    labels_processed: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def copy_ratio(self) -> Optional[float]:
        if self.total_files_in_stats > 0:
            return (self.copied_files / self.total_files_in_stats) * 100
        return None

    def to_dict(self) -> dict:
        return {
            "statistics_file": self.statistics_file,
            "source_dir": self.source_dir,
            "target_dir": self.target_dir,
            "total_files_in_stats": self.total_files_in_stats,
            "copied_files": self.copied_files,
            "skipped_files": self.skipped_files,
            "missing_files": self.missing_files,
            "copied_file_paths": self.copied_file_paths,
            "labels_processed": self.labels_processed,
            "copy_ratio": self.copy_ratio,
            "duration_seconds": self.duration,
        }


class StatisticsFileProcessor:

    def __init__(
        self,
        statistics_file: str,
        target_dir: str,
        preserve_structure: bool = True,
        copy_images: bool = True,
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        use_tqdm: bool = True,
        max_workers: int = 4,
    ):
        self.statistics_file = Path(statistics_file)
        self.target_dir = Path(target_dir)
        self.preserve_structure = preserve_structure
        self.copy_images = copy_images
        self.progress_callback = progress_callback
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self.max_workers = max_workers
        self._pbar = None

        self.logger = setup_progress_logging("StatisticsFileProcessor", log_file, log_level, self.use_tqdm)

    def _load_statistics_file(self) -> Optional[Dict]:
        """
        加载统计JSON文件

        Returns:
            Optional[Dict]: 解析后的数据，失败返回None
        """
        if not self.statistics_file.exists():
            self.logger.error(f"统计文件不存在: {self.statistics_file}")
            return None

        data = parse_json_file(self.statistics_file, self.logger)
        if data is None:
            self.logger.error("统计文件解析失败")
        return data

    def _extract_json_files(self, data: Dict) -> Set[str]:
        """
        从label_counts中提取所有JSON文件路径

        Args:
            data: 统计JSON数据

        Returns:
            Set[str]: JSON文件路径集合
        """
        json_files = set()

        label_counts = data.get("label_counts", {})
        if not label_counts:
            self.logger.warning("统计文件中没有label_counts数据")
            return json_files

        for label, count_dict in label_counts.items():
            for count, files in count_dict.items():
                if isinstance(files, list):
                    json_files.update(files)
                elif isinstance(files, set):
                    json_files.update(files)

        return json_files

    def _get_source_dir(self, data: Dict) -> Optional[Path]:
        """
        从统计数据中获取源目录

        Args:
            data: 统计JSON数据

        Returns:
            Optional[Path]: 源目录路径
        """
        source_dir = data.get("source_dir", "")
        if source_dir:
            return Path(source_dir)

        statistics_info = data.get("statistics_info", {})
        source_dir = statistics_info.get("source_dir", "")
        if source_dir:
            return Path(source_dir)

        return None

    def _copy_file(self, source_path: Path, target_dir: Path) -> Optional[str]:
        """
        复制文件到目标目录

        Args:
            source_path: 源文件路径
            target_dir: 目标目录

        Returns:
            Optional[str]: 复制后的文件路径，失败返回None
        """
        if self.preserve_structure:
            relative_path = source_path.name
            target_path = target_dir / relative_path
        else:
            target_path = target_dir / source_path.name

        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(source_path, target_path)
            return str(target_path)
        except Exception as e:
            self.logger.error(f"复制文件失败: {source_path} - {e}")
            return None

    def process(self) -> FilterCopyResult:
        """
        执行筛选复制处理

        Returns:
            FilterCopyResult: 处理结果对象
        """
        result = FilterCopyResult(statistics_file=str(self.statistics_file), source_dir="", target_dir=str(self.target_dir))
        result.start_time = datetime.now()

        self.target_dir.mkdir(parents=True, exist_ok=True)

        data = self._load_statistics_file()
        if data is None:
            result.end_time = datetime.now()
            return result

        source_dir = self._get_source_dir(data)
        if source_dir is None:
            self.logger.error("无法确定源目录")
            result.end_time = datetime.now()
            return result

        result.source_dir = str(source_dir)

        json_files = self._extract_json_files(data)
        result.total_files_in_stats = len(json_files)

        if not json_files:
            self.logger.warning("统计文件中没有找到JSON文件路径")
            result.end_time = datetime.now()
            return result

        self.logger.info(f"从统计文件中提取到 {len(json_files)} 个JSON文件")

        labels = data.get("label_counts", {}).keys()
        result.labels_processed = sorted(labels)

        sorted_files = sorted(json_files)

        if self.use_tqdm and not self.progress_callback:
            self._pbar = create_progress_bar(
                total=len(sorted_files),
                desc="筛选复制",
                unit="文件",
            )

        for i, json_file in enumerate(sorted_files, 1):
            self.logger.info(f"[{i}/{len(sorted_files)}] 处理: {json_file}")

            json_path = source_dir / json_file

            if not json_path.exists():
                result.skipped_files += 1
                result.missing_files.append(json_file)
                self.logger.warning(f"[{i}/{len(sorted_files)}] 文件不存在: {json_file}")
            else:
                copied_json = self._copy_file(json_path, self.target_dir)
                if copied_json:
                    result.copied_files += 1
                    result.copied_file_paths.append(copied_json)
                    self.logger.info(f"[{i}/{len(sorted_files)}] 已复制: {json_file}")

                if self.copy_images:
                    image_path = find_image_file(json_path)
                    if image_path:
                        copied_image = self._copy_file(image_path, self.target_dir)
                        if copied_image:
                            self.logger.info(f"[{i}/{len(sorted_files)}] 已复制图片: {image_path.name}")

            if self._pbar:
                self._pbar.update(1)
            elif self.progress_callback:
                self.progress_callback(Path(json_file).name, i, len(sorted_files))

        result.end_time = datetime.now()

        self.logger.info("=" * 50)
        self.logger.info("处理完成！")
        self.logger.info(f"统计文件中的JSON数: {result.total_files_in_stats}")
        self.logger.info(f"成功复制: {result.copied_files}")
        self.logger.info(f"跳过(不存在): {result.skipped_files}")
        self.logger.info(f"处理的类别数: {len(result.labels_processed)}")
        if result.copy_ratio:
            self.logger.info(f"复制率: {result.copy_ratio:.1f}%")
        if result.duration:
            self.logger.info(f"耗时: {result.duration:.2f} 秒")
        self.logger.info("=" * 50)

        return result


def process_statistics_file(
    statistics_file: str,
    target_dir: str,
    preserve_structure: bool = True,
    copy_images: bool = True,
    log_file: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    use_tqdm: bool = True,
    max_workers: int = 4,
) -> FilterCopyResult:
    processor = StatisticsFileProcessor(
        statistics_file=statistics_file,
        target_dir=target_dir,
        preserve_structure=preserve_structure,
        copy_images=copy_images,
        log_file=log_file,
        progress_callback=progress_callback,
        use_tqdm=use_tqdm,
        max_workers=max_workers,
    )

    return processor.process()


def process_main():
    """处理统计文件主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="处理统计JSON文件，筛选复制有效文件")
    parser.add_argument("statistics_file", help="统计JSON文件路径")
    parser.add_argument("-o", "--output", required=True, help="目标目录路径")
    parser.add_argument("--no-structure", action="store_true", help="不保留目录结构")
    parser.add_argument("--no-images", action="store_true", help="不复制图片文件")
    parser.add_argument("--log", help="日志文件路径")
    parser.add_argument("--report", help="生成处理报告文件路径")

    args = parser.parse_args()

    print("=" * 60)
    print("统计文件处理器")
    print("=" * 60)
    print(f"统计文件: {args.statistics_file}")
    print(f"目标目录: {args.output}")
    print(f"保留目录结构: {'否' if args.no_structure else '是'}")
    print(f"复制图片文件: {'否' if args.no_images else '是'}")
    print("-" * 60)

    def progress_callback(filename: str, current: int, total: int):
        percent = (current / total) * 100
        print(f"\r进度: [{current}/{total}] {percent:.1f}% - {filename}", end="", flush=True)

    result = process_statistics_file(
        statistics_file=args.statistics_file, target_dir=args.output, preserve_structure=not args.no_structure, copy_images=not args.no_images, log_file=args.log, progress_callback=progress_callback
    )

    print("\n")
    print("=" * 60)
    print("处理结果汇总")
    print("=" * 60)
    print(f"统计文件中的JSON数: {result.total_files_in_stats}")
    print(f"成功复制: {result.copied_files}")
    print(f"跳过(不存在): {result.skipped_files}")
    print(f"处理的类别数: {len(result.labels_processed)}")
    if result.copy_ratio:
        print(f"复制率: {result.copy_ratio:.1f}%")

    if result.missing_files:
        print("\n缺失文件列表:")
        for missing in result.missing_files[:10]:
            print(f"  - {missing}")
        if len(result.missing_files) > 10:
            print(f"  ... 还有 {len(result.missing_files) - 10} 个文件")

    if result.labels_processed:
        print(f"\n处理的类别: {', '.join(result.labels_processed)}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_file(report_path, result.to_dict(), indent=2)
        print(f"\n处理报告已保存到: {args.report}")

    if result.duration:
        print(f"\n总耗时: {result.duration:.2f} 秒")

    return result


if __name__ == "__main__":
    main()
