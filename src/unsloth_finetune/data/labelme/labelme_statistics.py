"""
LabelMe标注类别统计工具
统计LabelMe标注文件中的类别信息，生成类别统计报告
"""

import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .progress_logger import TQDM_AVAILABLE, setup_progress_logging, create_progress_bar
from .file_utils import find_json_files, parse_json_file, write_json_file


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

    def _has_image_reference(self, data: Dict) -> bool:
        """
        检查是否包含图片引用字段（imageUrl 或 imagePath）

        Args:
            data: JSON解析后的数据

        Returns:
            bool: 是否包含图片引用
        """
        has_image_url = "imageUrl" in data and data["imageUrl"]
        has_image_path = "imagePath" in data and data["imagePath"]
        return has_image_url or has_image_path

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
        result = {"json_path": str(json_path), "success": False, "has_image_ref": False, "label_counts": {}, "error_type": None}

        data = parse_json_file(json_path, self.logger)

        if data is None:
            result["error_type"] = "parse_error"
            with counter_lock:
                counter["skipped_parse_error"] += 1
                counter["skipped_files"] += 1
            return result

        if not self._has_image_reference(data):
            result["error_type"] = "no_image_ref"
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
        result["has_image_ref"] = True
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
                        elif process_result["error_type"] == "no_image_ref":
                            self.logger.debug(f"[{completed_count}/{len(json_files)}] 跳过(无图片引用): {json_path.name}")
                        elif process_result["error_type"] == "parse_error":
                            self.logger.debug(f"[{completed_count}/{len(json_files)}] 跳过(解析错误): {json_path.name}")
                    except Exception as e:
                        self.logger.warning(f"处理异常: {json_path} - {e}")
        else:
            for i, json_path in enumerate(json_files, 1):
                process_result = self._process_single_file(json_path, global_dict, global_lock, counter, counter_lock)

                if process_result["success"]:
                    self.logger.debug(f"[{i}/{len(json_files)}] 完成: {json_path.name} - " f"{len(process_result['label_counts'])} 个类别")
                elif process_result["error_type"] == "no_image_ref":
                    self.logger.debug(f"[{i}/{len(json_files)}] 跳过(无图片引用): {json_path.name}")
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
        self.logger.info(f"  - 无图片引用: {result.skipped_no_imageurl}")
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
    print(f"  - 无图片引用: {result.skipped_no_imageurl}")
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




if __name__ == "__main__":
    statistics_main()
