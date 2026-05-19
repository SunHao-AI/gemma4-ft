"""
LabelMe统计结果处理工具
根据统计结果JSON文件筛选并复制有效标注文件
"""

import logging
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .progress_logger import TQDM_AVAILABLE, setup_progress_logging, create_progress_bar
from .file_utils import (
    find_json_files,
    parse_json_file,
    write_json_file,
    find_image_file,
    create_file_link,
    ORJSON_AVAILABLE,
)


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
    process_main()