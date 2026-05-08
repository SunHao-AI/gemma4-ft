"""
共享文件操作工具模块
提取各模块中重复出现的JSON文件查找、解析、图片匹配等通用操作
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from .progress_logger import SUPPORTED_IMAGE_EXTENSIONS


def find_json_files(
    source_dir: Path,
    recursive: bool = True,
    logger: Optional[logging.Logger] = None,
) -> List[Path]:
    """
    查找目录下所有JSON文件

    Args:
        source_dir: 源目录路径
        recursive: 是否递归遍历子目录，默认True
        logger: 日志记录器，None则不输出日志

    Returns:
        List[Path]: JSON文件路径列表（已排序）
    """
    json_files = []

    if not source_dir.exists():
        if logger:
            logger.error(f"源目录不存在: {source_dir}")
        return json_files

    glob_func = source_dir.rglob if recursive else source_dir.glob
    for file_path in glob_func("*.json"):
        if file_path.is_file():
            json_files.append(file_path)

    if logger:
        logger.info(f"找到 {len(json_files)} 个JSON文件")

    return sorted(json_files)


def parse_json_file(
    json_path: Path,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict]:
    """
    解析JSON文件（支持utf-8/gbk编码自动回退）

    Args:
        json_path: JSON文件路径
        logger: 日志记录器，None则不输出日志

    Returns:
        Optional[Dict]: 解析后的数据字典，失败返回None
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        if logger:
            logger.warning(f"JSON解析错误: {json_path} - {e}")
        return None
    except UnicodeDecodeError:
        try:
            with open(json_path, "r", encoding="gbk") as f:
                return json.load(f)
        except Exception as e:
            if logger:
                logger.warning(f"编码错误: {json_path} - {e}")
            return None
    except Exception as e:
        if logger:
            logger.warning(f"文件读取错误: {json_path} - {e}")
        return None


def find_image_file(
    json_path: Path,
    image_path_str: Optional[str] = None,
    strict_name_match: bool = False,
    supported_extensions: Optional[Set[str]] = None,
) -> Optional[Path]:
    """
    查找JSON文件对应的图片文件

    支持三种匹配模式:
    - image_path_str=None: 仅按JSON文件名匹配图片扩展名（StatisticsFileProcessor模式）
    - image_path_str given + strict_name_match=True: 严格名称匹配（LabelMeCleaner模式）
    - image_path_str given + strict_name_match=False: 宽松候选路径匹配（LabelMeConverter/LabelMeSampler模式）

    Args:
        json_path: JSON文件路径
        image_path_str: JSON中的imagePath字段值，None则仅按文件名匹配扩展名
        strict_name_match: 是否严格匹配JSON文件名与图片文件名
        supported_extensions: 支持的图片扩展名集合，None使用默认值

    Returns:
        Optional[Path]: 图片文件路径，不存在则返回None
    """
    if supported_extensions is None:
        supported_extensions = SUPPORTED_IMAGE_EXTENSIONS

    json_dir = json_path.parent
    json_stem = json_path.stem

    if image_path_str is None:
        for ext in supported_extensions:
            candidate = json_dir / (json_stem + ext)
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    image_path = Path(image_path_str)

    if strict_name_match:
        if image_path.is_absolute():
            if image_path.exists() and image_path.is_file():
                if image_path.stem.lower() == json_stem.lower():
                    return image_path
            return None

        direct_path = json_dir / image_path_str
        if direct_path.exists() and direct_path.is_file():
            if direct_path.stem.lower() == json_stem.lower():
                return direct_path

        name_only_path = json_dir / image_path.name
        if name_only_path.exists() and name_only_path.is_file():
            if name_only_path.stem.lower() == json_stem.lower():
                return name_only_path

        for ext in supported_extensions:
            candidate = json_dir / (json_stem + ext)
            if candidate.exists() and candidate.is_file():
                return candidate

        return None

    candidate_paths = []

    if image_path.is_absolute():
        candidate_paths = [image_path]
    else:
        candidate_paths = [
            json_dir / image_path_str,
            json_dir / image_path.name,
            json_dir / (json_stem + image_path.suffix),
        ]
        for ext in supported_extensions:
            candidate_paths.append(json_dir / (json_stem + ext))

    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def get_relative_path(file_path: Path, base_dir: Path) -> Path:
    """
    获取文件相对于基准目录的相对路径

    Args:
        file_path: 文件路径
        base_dir: 基准目录路径

    Returns:
        Path: 相对路径，无法计算时返回文件名
    """
    try:
        return file_path.relative_to(base_dir)
    except ValueError:
        return Path(file_path.name)