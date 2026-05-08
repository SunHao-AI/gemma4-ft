"""
LabelMe标注数据样本均衡化选择工具
支持两种选择模式: n张图片模式和n个标签样本模式
"""

import json
import logging
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Set
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import threading

from .progress_logger import TQDM_AVAILABLE, setup_progress_logging, create_progress_bar
from .file_utils import find_json_files, parse_json_file, find_image_file

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class SelectionMode(Enum):
    """样本选择模式枚举"""

    N_IMAGES = "n_images"
    N_LABELS = "n_labels"


@dataclass
class ImageLabelInfo:
    """图像标签信息数据类"""

    json_path: str
    image_path: Optional[str] = None
    label_counts: Dict[str, int] = field(default_factory=dict)
    total_labels: int = 0

    def __post_init__(self):
        self.total_labels = sum(self.label_counts.values())


@dataclass
class SelectionResult:
    """样本选择结果数据类"""

    category: str
    mode: SelectionMode
    target_count: int
    selected_images: List[ImageLabelInfo] = field(default_factory=list)
    total_selected_images: int = 0
    total_selected_labels: int = 0
    has_duplicates: bool = False
    duplicate_count: int = 0
    available_images: int = 0
    available_labels: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "mode": self.mode.value,
            "target_count": self.target_count,
            "total_selected_images": self.total_selected_images,
            "total_selected_labels": self.total_selected_labels,
            "has_duplicates": self.has_duplicates,
            "duplicate_count": self.duplicate_count,
            "available_images": self.available_images,
            "available_labels": self.available_labels,
            "selected_images": [{"json_path": img.json_path, "image_path": img.image_path, "label_counts": img.label_counts, "total_labels": img.total_labels} for img in self.selected_images],
            "duration_seconds": self.duration,
        }


@dataclass
class BalancedSelectionResult:
    """均衡选择结果数据类"""

    source_dir: str
    mode: SelectionMode
    target_count: int
    random_seed: Optional[int] = None
    category_results: Dict[str, SelectionResult] = field(default_factory=dict)
    total_selected_images: int = 0
    unique_images: Set[str] = field(default_factory=set)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def unique_image_count(self) -> int:
        return len(self.unique_images)

    def to_dict(self) -> dict:
        return {
            "source_dir": self.source_dir,
            "mode": self.mode.value,
            "target_count": self.target_count,
            "random_seed": self.random_seed,
            "total_selected_images": self.total_selected_images,
            "unique_image_count": self.unique_image_count,
            "category_results": {cat: result.to_dict() for cat, result in self.category_results.items()},
            "duration_seconds": self.duration,
        }


class LabelMeSampler:
    """LabelMe样本均衡化选择工具类"""

    def __init__(
        self,
        source_dir: str,
        mode: SelectionMode = SelectionMode.N_IMAGES,
        target_count: int = 100,
        random_seed: Optional[int] = None,
        validate_images: bool = True,
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        use_tqdm: bool = True,
    ):
        """
        初始化样本选择工具

        Args:
            source_dir: 源目录路径
            mode: 选择模式 (n_images 或 n_labels)
            target_count: 目标数量(n张图片或n个标签样本)
            random_seed: 随机种子
            validate_images: 是否验证图片文件存在
            log_file: 日志文件路径
            log_level: 日志级别
            max_workers: 最大线程数
            progress_callback: 进度回调函数
            use_tqdm: 是否使用tqdm进度条，默认True
        """
        self.source_dir = Path(source_dir)
        self.mode = mode
        self.target_count = target_count
        self.random_seed = random_seed
        self.validate_images = validate_images
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self._pbar = None

        self.logger = setup_progress_logging("LabelMeSampler", log_file, log_level, self.use_tqdm)

        if random_seed is not None:
            random.seed(random_seed)

    def _extract_label_counts(self, data: Dict) -> Dict[str, int]:
        """提取标签数量统计"""
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

    def _process_single_file(self, json_path: Path, global_dict: Dict[str, List[ImageLabelInfo]], lock: threading.Lock) -> Optional[ImageLabelInfo]:
        """处理单个JSON文件"""
        data = parse_json_file(json_path, self.logger)
        if data is None:
            return None

        label_counts = self._extract_label_counts(data)
        if not label_counts:
            return None

        image_path = None
        if self.validate_images:
            image_path_str = data.get("imagePath", "")
            if image_path_str:
                image_path = find_image_file(json_path, image_path_str)
                if image_path is None:
                    self.logger.warning(f"图片不存在: {json_path}")
                    return None

        image_info = ImageLabelInfo(json_path=str(json_path), image_path=str(image_path) if image_path else None, label_counts=label_counts)

        with lock:
            for label in label_counts.keys():
                if label not in global_dict:
                    global_dict[label] = []
                global_dict[label].append(image_info)

        return image_info

    def _build_category_image_map(self) -> Dict[str, List[ImageLabelInfo]]:
        """构建类别到图像的映射"""
        json_files = find_json_files(self.source_dir, logger=self.logger)

        if not json_files:
            return {}

        if self.use_tqdm and not self.progress_callback:
            self._pbar = create_progress_bar(
                total=len(json_files),
                desc="样本选择",
                unit="文件",
            )

        self.logger.info(f"开始处理 {len(json_files)} 个JSON文件")

        global_dict: Dict[str, List[ImageLabelInfo]] = {}
        global_lock = threading.Lock()

        processed_count = 0

        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._process_single_file, json_path, global_dict, global_lock): json_path for json_path in json_files}

                for future in as_completed(futures):
                    processed_count += 1
                    json_path = futures[future]

                    if self._pbar:
                        self._pbar.update(1)
                    elif self.progress_callback:
                        self.progress_callback(json_path.name, processed_count, len(json_files))

                    try:
                        future.result()
                    except Exception as e:
                        self.logger.warning(f"[处理] 错误: {json_path.name} - {e}")
        else:
            for i, json_path in enumerate(json_files, 1):
                if self._pbar:
                    self._pbar.update(1)
                elif self.progress_callback:
                    self.progress_callback(json_path.name, i, len(json_files))

                self._process_single_file(json_path, global_dict, global_lock)

        if self._pbar:
            self._pbar.close()
            self._pbar = None

        for label in global_dict:
            global_dict[label].sort(key=lambda x: x.total_labels, reverse=True)

        return global_dict

    def _select_n_images(self, images: List[ImageLabelInfo], n: int) -> SelectionResult:
        """n张图片模式选择"""
        result = SelectionResult(category="", mode=SelectionMode.N_IMAGES, target_count=n)
        result.start_time = datetime.now()
        result.available_images = len(images)
        result.available_labels = sum(img.total_labels for img in images)

        if len(images) <= n:
            selected = images.copy()
            result.has_duplicates = True

            cycles_needed = (n // len(images)) + (1 if n % len(images) > 0 else 0)

            for _ in range(cycles_needed - 1):
                selected.extend(images.copy())

            if n % len(images) > 0:
                selected.extend(images[: n % len(images)])

            result.duplicate_count = len(selected) - len(images)
        else:
            selected = images[:n]

        result.selected_images = selected[:n]
        result.total_selected_images = len(result.selected_images)
        result.total_selected_labels = sum(img.total_labels for img in result.selected_images)
        result.end_time = datetime.now()

        return result

    def _select_n_labels(self, images: List[ImageLabelInfo], n: int, category: str) -> SelectionResult:
        """n个标签样本模式选择"""
        result = SelectionResult(category=category, mode=SelectionMode.N_LABELS, target_count=n)
        result.start_time = datetime.now()
        result.available_images = len(images)
        result.available_labels = sum(img.label_counts.get(category, 0) for img in images)

        total_label_count = 0
        selected = []

        for img in images:
            label_count = img.label_counts.get(category, 0)
            if total_label_count >= n:
                break

            selected.append(img)
            total_label_count += label_count

        if total_label_count < n:
            result.has_duplicates = True

            while total_label_count < n:
                for img in images:
                    label_count = img.label_counts.get(category, 0)
                    if total_label_count >= n:
                        break

                    selected.append(img)
                    total_label_count += label_count

            result.duplicate_count = len(selected) - len(set(img.json_path for img in selected))

        result.selected_images = selected
        result.total_selected_images = len(selected)
        result.total_selected_labels = total_label_count
        result.end_time = datetime.now()

        return result

    def select_samples(self) -> BalancedSelectionResult:
        """执行样本选择"""
        result = BalancedSelectionResult(source_dir=str(self.source_dir), mode=self.mode, target_count=self.target_count, random_seed=self.random_seed)
        result.start_time = datetime.now()

        category_image_map = self._build_category_image_map()

        if not category_image_map:
            self.logger.warning("未找到任何有效的标注数据")
            result.end_time = datetime.now()
            return result

        self.logger.info(f"找到 {len(category_image_map)} 个类别")

        for category, images in category_image_map.items():
            self.logger.info(f"处理类别: {category} ({len(images)} 张图片)")

            if self.mode == SelectionMode.N_IMAGES:
                selection_result = self._select_n_images(images, self.target_count)
            else:
                selection_result = self._select_n_labels(images, self.target_count, category)

            selection_result.category = category
            result.category_results[category] = selection_result
            result.total_selected_images += selection_result.total_selected_images

            for img in selection_result.selected_images:
                result.unique_images.add(img.json_path)

        result.end_time = datetime.now()

        self.logger.info("=" * 50)
        self.logger.info("样本选择完成！")
        self.logger.info(f"类别总数: {len(result.category_results)}")
        self.logger.info(f"总选择图片数: {result.total_selected_images}")
        self.logger.info(f"唯一图片数: {result.unique_image_count}")
        if result.duration:
            self.logger.info(f"耗时: {result.duration:.2f} 秒")
        self.logger.info("=" * 50)

        return result

    def get_selected_files(self) -> List[str]:
        """获取选择的文件路径列表"""
        result = self.select_samples()
        return sorted(result.unique_images)


def select_balanced_samples(
    source_dir: str,
    mode: str = "n_images",
    target_count: int = 100,
    random_seed: Optional[int] = None,
    validate_images: bool = True,
    log_file: Optional[str] = None,
    max_workers: int = 4,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    use_tqdm: bool = True,
) -> BalancedSelectionResult:
    """
    均衡选择LabelMe标注样本

    Args:
        source_dir: 源目录路径
        mode: 选择模式 ("n_images" 或 "n_labels")
        target_count: 目标数量
        random_seed: 随机种子
        validate_images: 是否验证图片文件存在
        log_file: 日志文件路径
        max_workers: 最大线程数
        progress_callback: 进度回调函数
        use_tqdm: 是否使用tqdm进度条，默认True

    Returns:
        BalancedSelectionResult: 选择结果对象

    Example:
        >>> result = select_balanced_samples(
        ...     source_dir="path/to/data",
        ...     mode="n_images",
        ...     target_count=50
        ... )
        >>> print(f"选择了 {result.unique_image_count} 张唯一图片")
    """
    selection_mode = SelectionMode.N_IMAGES if mode == "n_images" else SelectionMode.N_LABELS

    sampler = LabelMeSampler(
        source_dir=source_dir,
        mode=selection_mode,
        target_count=target_count,
        random_seed=random_seed,
        validate_images=validate_images,
        log_file=log_file,
        max_workers=max_workers,
        progress_callback=progress_callback,
        use_tqdm=use_tqdm,
    )

    return sampler.select_samples()


def main():
    """主函数示例"""
    import argparse

    parser = argparse.ArgumentParser(description="LabelMe样本均衡化选择工具")
    parser.add_argument("source", help="源目录路径")
    parser.add_argument("--mode", choices=["n_images", "n_labels"], default="n_images", help="选择模式")
    parser.add_argument("--count", type=int, default=100, help="目标数量")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--workers", "-w", type=int, default=4, help="最大线程数")
    parser.add_argument("--log", help="日志文件路径")
    parser.add_argument("--output", "-o", help="输出JSON文件路径")

    args = parser.parse_args()

    print("=" * 60)
    print("LabelMe样本均衡化选择工具")
    print("=" * 60)
    print(f"源目录: {args.source}")
    print(f"选择模式: {args.mode}")
    print(f"目标数量: {args.count}")
    print(f"随机种子: {args.seed if args.seed else '无'}")
    print("-" * 60)

    def progress_callback(filename: str, current: int, total: int):
        percent = (current / total) * 100
        print(f"\r进度: [{current}/{total}] {percent:.1f}% - {filename}", end="", flush=True)

    result = select_balanced_samples(
        source_dir=args.source, mode=args.mode, target_count=args.count, random_seed=args.seed, log_file=args.log, max_workers=args.workers, progress_callback=progress_callback
    )

    print("\n")
    print("=" * 60)
    print("选择结果汇总")
    print("=" * 60)
    print(f"类别总数: {len(result.category_results)}")
    print(f"总选择图片数: {result.total_selected_images}")
    print(f"唯一图片数: {result.unique_image_count}")

    for category, sel_result in sorted(result.category_results.items()):
        print(f"\n类别: {category}")
        print(f"  选择图片数: {sel_result.total_selected_images}")
        print(f"  选择标签数: {sel_result.total_selected_labels}")
        print(f"  可用图片数: {sel_result.available_images}")
        print(f"  可用标签数: {sel_result.available_labels}")
        print(f"  是否重复: {'是' if sel_result.has_duplicates else '否'}")
        if sel_result.has_duplicates:
            print(f"  重复数量: {sel_result.duplicate_count}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n选择结果已保存到: {args.output}")

    if result.duration:
        print(f"\n总耗时: {result.duration:.2f} 秒")

    return result


if __name__ == "__main__":
    main()
