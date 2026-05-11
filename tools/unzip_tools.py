"""
压缩文件解压工具
支持多种压缩格式：.zip, .rar, .7z, .tar, .tar.gz, .tgz, .tar.bz2, .tbz2
"""

import zipfile
import tarfile
import logging
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .progress_logger import (
    TQDM_AVAILABLE,
    setup_progress_logging,
    create_progress_bar,
)

try:
    import rarfile

    RARFILE_AVAILABLE = True
except ImportError:
    RARFILE_AVAILABLE = False

try:
    import py7zr

    PY7ZR_AVAILABLE = True
except ImportError:
    PY7ZR_AVAILABLE = False


@dataclass
class UnzipResult:
    """解压结果数据类"""

    total_files: int = 0
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    success_files: List[str] = field(default_factory=list)
    failed_files: List[Dict[str, str]] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> dict:
        return {
            "total_files": self.total_files,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "success_files": self.success_files,
            "failed_files": self.failed_files,
            "skipped_files": self.skipped_files,
            "duration_seconds": self.duration,
        }


class UnzipTool:
    """压缩文件解压工具类"""

    SUPPORTED_EXTENSIONS = {
        ".zip": "zip",
        ".rar": "rar",
        ".7z": "7z",
        ".tar": "tar",
        ".tar.gz": "tar",
        ".tgz": "tar",
        ".tar.bz2": "tar",
        ".tbz2": "tar",
    }

    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        overwrite: bool = False,
        rename_on_conflict: bool = True,
        simplify_path: bool = True,
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        max_workers: int = 1,
        chunk_size: int = 1024 * 1024 * 10,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        use_tqdm: bool = True,
    ):
        """
        初始化解压工具

        Args:
            source_dir: 源目录路径
            target_dir: 目标目录路径
            overwrite: 是否覆盖已存在的文件
            rename_on_conflict: 文件重名时是否重命名（仅在overwrite=False时生效）
            simplify_path: 是否简化重复的父文件夹路径，默认True
            log_file: 日志文件路径，None则不记录到文件
            log_level: 日志级别
            max_workers: 最大并发解压数（建议保持1以避免磁盘IO竞争）
            chunk_size: 解压时读取块大小（字节），用于控制内存占用
            progress_callback: 进度回调函数，参数为(文件名, 当前索引, 总数)
            use_tqdm: 是否使用tqdm进度条，默认True
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.overwrite = overwrite
        self.rename_on_conflict = rename_on_conflict
        self.simplify_path = simplify_path
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.progress_callback = progress_callback
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self._pbar = None

        self.logger = setup_progress_logging("UnzipTool", log_file, log_level, self.use_tqdm)
        self._check_dependencies()

    def _check_dependencies(self):
        """检查依赖库"""
        if not RARFILE_AVAILABLE:
            self.logger.warning("rarfile库未安装，将无法解压.rar文件。请运行: pip install rarfile")
        if not PY7ZR_AVAILABLE:
            self.logger.warning("py7zr库未安装，将无法解压.7z文件。请运行: pip install py7zr")

    def _get_archive_type(self, file_path: Path) -> Optional[str]:
        """获取压缩文件类型"""
        name_lower = file_path.name.lower()
        for ext, archive_type in self.SUPPORTED_EXTENSIONS.items():
            if name_lower.endswith(ext):
                return archive_type
        return None

    def _simplify_duplicate_path(self, extract_dir: Path) -> bool:
        """
        简化重复的父文件夹路径
        如果解压目录下存在一个同名子文件夹，将其内容提升到解压目录根层级

        Args:
            extract_dir: 解压目录路径

        Returns:
            bool: 是否进行了路径简化
        """
        if not extract_dir.exists() or not extract_dir.is_dir():
            return False

        items = list(extract_dir.iterdir())

        if len(items) != 1:
            return False

        single_item = items[0]

        if not single_item.is_dir():
            return False

        extract_dir_name = extract_dir.name.lower().replace("_", "").replace(" ", "")
        single_item_name = single_item.name.lower().replace("_", "").replace(" ", "")

        if single_item_name != extract_dir_name:
            return False

        self.logger.info(f"检测到重复路径结构，正在简化: {single_item.name}")

        try:
            temp_dir = extract_dir.parent / f"_temp_{extract_dir.name}"

            shutil.move(str(single_item), str(temp_dir))

            for item in temp_dir.iterdir():
                target = extract_dir / item.name
                if target.exists():
                    if self.overwrite:
                        if target.is_dir():
                            shutil.rmtree(str(target))
                        else:
                            target.unlink()
                    elif self.rename_on_conflict:
                        target = self._get_unique_path(target)
                    else:
                        continue

                shutil.move(str(item), str(target))

            shutil.rmtree(str(temp_dir))

            self.logger.info(f"路径简化完成: {extract_dir}")
            return True

        except Exception as e:
            self.logger.error(f"路径简化失败: {e}")
            return False

    def _ensure_target_dir(self):
        """确保目标目录存在"""
        if not self.target_dir.exists():
            self.target_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"创建目标目录: {self.target_dir}")

    def _get_unique_path(self, target_path: Path) -> Path:
        """获取唯一路径（处理重名文件）"""
        if not target_path.exists():
            return target_path

        if self.overwrite:
            return target_path

        if not self.rename_on_conflict:
            return target_path

        parent = target_path.parent
        stem = target_path.stem
        suffix = target_path.suffix
        counter = 1

        while True:
            new_name = f"{stem}_{counter}{suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1

    def _extract_zip(self, archive_path: Path, extract_to: Path) -> bool:
        """解压ZIP文件"""
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                for member in zf.namelist():
                    member_path = extract_to / member
                    if member.endswith("/"):
                        member_path.mkdir(parents=True, exist_ok=True)
                    else:
                        member_path.parent.mkdir(parents=True, exist_ok=True)
                        final_path = self._get_unique_path(member_path)
                        with zf.open(member) as src, open(final_path, "wb") as dst:
                            while True:
                                chunk = src.read(self.chunk_size)
                                if not chunk:
                                    break
                                dst.write(chunk)
            return True
        except zipfile.BadZipFile as e:
            raise ValueError(f"ZIP文件损坏: {e}")
        except PermissionError as e:
            raise PermissionError(f"权限不足: {e}")

    def _extract_rar(self, archive_path: Path, extract_to: Path) -> bool:
        """解压RAR文件"""
        if not RARFILE_AVAILABLE:
            raise ImportError("rarfile库未安装，无法解压RAR文件")

        try:
            with rarfile.RarFile(archive_path, "r") as rf:
                for member in rf.namelist():
                    member_path = extract_to / member
                    if member.endswith("/"):
                        member_path.mkdir(parents=True, exist_ok=True)
                    else:
                        member_path.parent.mkdir(parents=True, exist_ok=True)
                        final_path = self._get_unique_path(member_path)
                        with rf.open(member) as src, open(final_path, "wb") as dst:
                            while True:
                                chunk = src.read(self.chunk_size)
                                if not chunk:
                                    break
                                dst.write(chunk)
            return True
        except rarfile.BadRarFile as e:
            raise ValueError(f"RAR文件损坏: {e}")
        except rarfile.NeedFirstVolume as e:
            raise ValueError(f"需要分卷的第一个文件: {e}")

    def _extract_7z(self, archive_path: Path, extract_to: Path) -> bool:
        """解压7z文件"""
        if not PY7ZR_AVAILABLE:
            raise ImportError("py7zr库未安装，无法解压7z文件")

        try:
            with py7zr.SevenZipFile(archive_path, mode="r") as szf:
                szf.extractall(path=str(extract_to))
            return True
        except py7zr.exceptions.Bad7zFile as e:
            raise ValueError(f"7z文件损坏: {e}")

    def _extract_tar(self, archive_path: Path, extract_to: Path) -> bool:
        """解压TAR文件"""
        try:
            with tarfile.open(archive_path, "r:*") as tf:
                for member in tf.getmembers():
                    member_path = extract_to / member.name
                    if member.isdir():
                        member_path.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        member_path.parent.mkdir(parents=True, exist_ok=True)
                        final_path = self._get_unique_path(member_path)
                        with tf.extractfile(member) as src, open(final_path, "wb") as dst:
                            while True:
                                chunk = src.read(self.chunk_size)
                                if not chunk:
                                    break
                                dst.write(chunk)
            return True
        except tarfile.TarError as e:
            raise ValueError(f"TAR文件损坏: {e}")

    def _extract_single_file(self, archive_path: Path, index: int, total: int) -> Dict:
        """解压单个压缩文件"""
        result = {"file": str(archive_path), "success": False, "error": None}

        try:
            archive_type = self._get_archive_type(archive_path)
            if not archive_type:
                result["error"] = f"不支持的压缩格式: {archive_path.suffix}"
                return result

            extract_to = self.target_dir / archive_path.stem
            extract_to.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"[{index}/{total}] 正在解压: {archive_path.name}")

            if self.progress_callback:
                self.progress_callback(archive_path.name, index, total)

            if archive_type == "zip":
                self._extract_zip(archive_path, extract_to)
            elif archive_type == "rar":
                self._extract_rar(archive_path, extract_to)
            elif archive_type == "7z":
                self._extract_7z(archive_path, extract_to)
            elif archive_type == "tar":
                self._extract_tar(archive_path, extract_to)

            if self.simplify_path:
                self._simplify_duplicate_path(extract_to)

            result["success"] = True

        except Exception as e:
            result["error"] = str(e)
            self.logger.warning(f"[解压] 失败: {archive_path.name} - {e}")

        return result

    def find_archive_files(self) -> List[Path]:
        """查找源目录下所有压缩文件"""
        archive_files = []

        if not self.source_dir.exists():
            self.logger.error(f"源目录不存在: {self.source_dir}")
            return archive_files

        for file_path in self.source_dir.rglob("*"):
            if file_path.is_file() and self._get_archive_type(file_path):
                archive_files.append(file_path)

        self.logger.info(f"找到 {len(archive_files)} 个压缩文件")
        return sorted(archive_files)

    def extract_all(self) -> UnzipResult:
        """解压所有压缩文件"""
        result = UnzipResult()
        result.start_time = datetime.now()

        self._ensure_target_dir()
        archive_files = self.find_archive_files()
        result.total_files = len(archive_files)

        if not archive_files:
            self.logger.warning("未找到任何压缩文件")
            result.end_time = datetime.now()
            return result

        total = len(archive_files)

        if self.use_tqdm and not self.progress_callback:
            self._pbar = create_progress_bar(
                total=total,
                desc="文件解压",
                unit="文件",
            )

        self.logger.info(f"开始解压 {total} 个压缩文件")

        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._extract_single_file, f, i + 1, total): f for i, f in enumerate(archive_files)}

                completed_count = 0
                for future in as_completed(futures):
                    completed_count += 1
                    extract_result = future.result()

                    if self._pbar:
                        self._pbar.update(1)

                    if extract_result["success"]:
                        result.success_count += 1
                        result.success_files.append(extract_result["file"])
                    else:
                        result.failed_count += 1
                        result.failed_files.append({"file": extract_result["file"], "reason": extract_result["error"]})
                        self.logger.warning(f"[解压] 失败: {Path(extract_result['file']).name}")
        else:
            for i, archive_path in enumerate(archive_files, 1):
                extract_result = self._extract_single_file(archive_path, i, total)

                if extract_result["success"]:
                    result.success_count += 1
                    result.success_files.append(extract_result["file"])
                else:
                    result.failed_count += 1
                    result.failed_files.append({"file": extract_result["file"], "reason": extract_result["error"]})

                if self._pbar:
                    self._pbar.update(1)
                elif self.progress_callback:
                    self.progress_callback(archive_path.name, i, total)

        if self._pbar:
            self._pbar.close()
            self._pbar = None

        result.end_time = datetime.now()

        self.logger.info("=" * 50)
        self.logger.info(f"解压完成！")
        self.logger.info(f"总文件数: {result.total_files}")
        self.logger.info(f"成功: {result.success_count}")
        self.logger.info(f"失败: {result.failed_count}")
        if result.duration:
            self.logger.info(f"耗时: {result.duration:.2f} 秒")
        self.logger.info("=" * 50)

        return result


def unzip_files(
    source_dir: str,
    target_dir: str,
    overwrite: bool = False,
    rename_on_conflict: bool = True,
    simplify_path: bool = True,
    log_file: Optional[str] = None,
    max_workers: int = 1,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    use_tqdm: bool = True,
) -> UnzipResult:
    """
    解压指定源目录下的所有压缩文件到目标目录

    Args:
        source_dir: 源目录路径
        target_dir: 目标目录路径
        overwrite: 是否覆盖已存在的文件，默认False
        rename_on_conflict: 文件重名时是否自动重命名，默认True
        simplify_path: 是否简化重复的父文件夹路径，默认True
        log_file: 日志文件路径，None则不记录到文件
        max_workers: 最大并发解压数，默认1（建议保持1避免磁盘IO竞争）
        progress_callback: 进度回调函数，参数为(文件名, 当前索引, 总数)
        use_tqdm: 是否使用tqdm进度条，默认True

    Returns:
        UnzipResult: 解压结果对象，包含成功/失败统计信息

    Example:
        >>> result = unzip_files()
        >>> print(f"成功解压 {result.success_count} 个文件")
        >>> print(f"失败 {result.failed_count} 个文件")
        >>> for failed in result.failed_files:
        ...     print(f"失败文件: {failed['file']}, 原因: {failed['reason']}")
    """
    tool = UnzipTool(
        source_dir=source_dir,
        target_dir=target_dir,
        overwrite=overwrite,
        rename_on_conflict=rename_on_conflict,
        simplify_path=simplify_path,
        log_file=log_file,
        max_workers=max_workers,
        progress_callback=progress_callback,
        use_tqdm=use_tqdm,
    )

    return tool.extract_all()


def main():
    """主函数示例"""
    print("=" * 60)
    print("压缩文件解压工具")
    print("=" * 60)

    source_dir = r"/raid5/sh/data/102_inference_zip"
    target_dir = r"/raid5/sh/data/102_inference"
    log_file = r"/raid5/sh/data/102_inference-unzip_log.txt"

    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"日志文件: {log_file}")
    print("-" * 60)

    def progress_callback(filename: str, current: int, total: int):
        percent = (current / total) * 100
        print(f"\r进度: [{current}/{total}] {percent:.1f}% - {filename}", end="", flush=True)

    result = unzip_files(source_dir=source_dir, target_dir=target_dir, log_file=log_file, progress_callback=progress_callback)

    print("\n")
    print("=" * 60)
    print("解压结果汇总")
    print("=" * 60)
    print(f"总文件数: {result.total_files}")
    print(f"成功解压: {result.success_count}")
    print(f"解压失败: {result.failed_count}")

    if result.failed_files:
        print("\n失败文件列表:")
        for failed in result.failed_files:
            print(f"  - {failed['file']}")
            print(f"    原因: {failed['reason']}")

    if result.duration:
        print(f"\n总耗时: {result.duration:.2f} 秒")

    return result


if __name__ == "__main__":
    main()
