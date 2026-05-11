"""
共享文件操作工具模块
提取各模块中重复出现的JSON文件查找、解析、图片匹配等通用操作

JSON性能优化: 优先使用orjson（Rust实现，3-10x faster），不可用时回退至stdlib json

跨平台文件链接: 提供符号链接/硬链接/复制的跨平台兼容实现
"""

import json
import logging
import os
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

from .progress_logger import SUPPORTED_IMAGE_EXTENSIONS

LinkType = Literal["auto", "symlink", "hardlink", "copy"]

try:
    import orjson

    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False


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


def _log_parse_error(
    json_path: Path,
    error_type: str,
    error: Exception,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    统一处理JSON解析错误的日志记录

    Args:
        json_path: JSON文件路径
        error_type: 错误类型描述（如"JSON解析错误"、"编码错误"等）
        error: 异常对象
        logger: 日志记录器，None则不输出日志
    """
    if logger:
        logger.warning(f"{error_type}: {json_path} - {error}")


def _try_parse_with_encoding_fallback(
    raw_bytes: bytes,
    json_path: Path,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict]:
    """
    尝试使用多种编码解析JSON字节数据

    Args:
        raw_bytes: 文件的原始字节数据
        json_path: JSON文件路径（用于日志）
        logger: 日志记录器

    Returns:
        Optional[Dict]: 解析成功返回字典，失败返回None
    """
    try:
        text_utf8 = raw_bytes.decode("utf-8")
        return json.loads(text_utf8)
    except json.JSONDecodeError as e:
        _log_parse_error(json_path, "JSON解析错误", e, logger)
        return None
    except UnicodeDecodeError:
        try:
            text_gbk = raw_bytes.decode("gbk")
            return json.loads(text_gbk)
        except Exception as e:
            _log_parse_error(json_path, "编码错误", e, logger)
            return None
    except Exception as e:
        _log_parse_error(json_path, "文件读取错误", e, logger)
        return None


def parse_json_file(
    json_path: Path,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict]:
    """
    解析JSON文件（支持utf-8/gbk编码自动回退）

    性能优化: 优先使用orjson.loads()解析bytes，约3-10x faster
    orjson仅支持UTF-8/UTF-16/UTF-32，GBK编码需回退至stdlib json

    Args:
        json_path: JSON文件路径
        logger: 日志记录器，None则不输出日志

    Returns:
        Optional[Dict]: 解析后的数据字典，失败返回None
    """
    if ORJSON_AVAILABLE:
        try:
            with open(json_path, "rb") as f:
                return orjson.loads(f.read())
        except orjson.JSONDecodeError as e:
            _log_parse_error(json_path, "JSON解析错误(orjson)", e, logger)
            return None
        except UnicodeDecodeError:
            pass
        except Exception:
            pass

        raw_bytes = json_path.read_bytes()
        return _try_parse_with_encoding_fallback(raw_bytes, json_path, logger)
    else:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            _log_parse_error(json_path, "JSON解析错误", e, logger)
            return None
        except UnicodeDecodeError:
            try:
                with open(json_path, "r", encoding="gbk") as f:
                    return json.load(f)
            except Exception as e:
                _log_parse_error(json_path, "编码错误", e, logger)
                return None
        except Exception as e:
            _log_parse_error(json_path, "文件读取错误", e, logger)
            return None


def json_loads(data: Union[str, bytes]) -> Dict:
    """
    解析JSON字符串/字节（优先使用orjson加速）

    Args:
        data: JSON格式的字符串或bytes

    Returns:
        Dict: 解析后的数据字典
    """
    if ORJSON_AVAILABLE:
        return orjson.loads(data)
    return json.loads(data)


def json_dumps_str(obj: object, indent: Optional[int] = None, ensure_ascii: bool = False) -> str:
    """
    序列化为JSON字符串（优先使用orjson加速）

    Args:
        obj: 要序列化的对象
        indent: 缩进级别，None为紧凑输出，2为标准缩进
        ensure_ascii: 是否确保ASCII输出（orjson默认输出UTF-8，此参数仅影响stdlib回退）

    Returns:
        str: JSON字符串
    """
    if ORJSON_AVAILABLE:
        if indent == 2:
            return orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode("utf-8")
        return orjson.dumps(obj).decode("utf-8")
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii)


def write_json_file(
    file_path: Path,
    data: object,
    indent: Optional[int] = None,
    ensure_ascii: bool = False,
) -> None:
    """
    将数据写入JSON文件（优先使用orjson加速）

    Args:
        file_path: 目标文件路径
        data: 要序列化的数据
        indent: 缩进级别，None为紧凑输出，2为标准缩进
        ensure_ascii: 是否确保ASCII输出（orjson默认UTF-8，此参数仅影响stdlib回退）
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if ORJSON_AVAILABLE:
        if indent == 2:
            content = orjson.dumps(data, option=orjson.OPT_INDENT_2)
        else:
            content = orjson.dumps(data)
        with open(file_path, "wb") as f:
            f.write(content)
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def find_image_file(
    json_path: Path,
    image_path_str: Optional[str] = None,
    strict_name_match: bool = False,
    supported_extensions: Optional[Set[str]] = None,
) -> Optional[Path]:
    """
    查找JSON文件对应的图片文件
    
    重要规范: 图片查找严格限定在JSON文件所在目录,不会跨目录查找
    
    查找优先级(均在JSON所在目录内):
    1. 如果imagePath字段存在,优先按imagePath指定的文件名查找
    2. 如果imagePath指定的文件不存在,尝试JSON文件名+图片扩展名
    3. 如果imagePath字段不存在,直接尝试JSON文件名+图片扩展名
    
    Args:
        json_path: JSON文件路径
        image_path_str: JSON中的imagePath字段值,None则仅按JSON文件名匹配
        strict_name_match: 是否要求图片文件名与JSON文件名一致(忽略扩展名)
        supported_extensions: 支持的图片扩展名集合,None使用默认值
    
    Returns:
        Optional[Path]: 图片文件路径(仅在JSON所在目录内),不存在则返回None
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
    
    if image_path.is_absolute():
        if image_path.exists() and image_path.is_file():
            if strict_name_match:
                if image_path.stem.lower() == json_stem.lower():
                    return image_path
            else:
                return image_path
        return None
    
    image_filename = image_path.name
    
    candidate1 = json_dir / image_filename
    if candidate1.exists() and candidate1.is_file():
        if strict_name_match:
            if candidate1.stem.lower() == json_stem.lower():
                return candidate1
        else:
            return candidate1
    
    for ext in supported_extensions:
        candidate2 = json_dir / (json_stem + ext)
        if candidate2.exists() and candidate2.is_file():
            return candidate2
    
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


def create_file_link(
    source_path: Path,
    target_path: Path,
    link_type: LinkType = "auto",
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """
    创建跨平台文件链接（符号链接/硬链接/复制）

    Windows平台兼容性策略：
    1. 首先尝试符号链接(symlink)，Windows 10+支持但可能需要管理员权限
    2. 如果符号链接失败，尝试硬链接(hardlink)，仅用于同一文件系统的文件
    3. 如果硬链接也失败，回退到普通文件复制

    macOS/Linux平台：
    - 直接使用符号链接，无需特殊处理

    Args:
        source_path: 源文件路径
        target_path: 目标链接/复制路径
        link_type: 链接类型
            - 'auto': 自动选择最佳方式（推荐）
            - 'symlink': 强制使用符号链接
            - 'hardlink': 强制使用硬链接
            - 'copy': 强制使用复制
        logger: 日志记录器

    Returns:
        Tuple[bool, str]: (是否成功, 实际使用的方法描述)
    """
    if not source_path.exists():
        msg = f"源文件不存在: {source_path}"
        if logger:
            logger.error(msg)
        return False, msg

    if not source_path.is_file():
        msg = f"源路径不是文件: {source_path}"
        if logger:
            logger.error(msg)
        return False, msg

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        try:
            target_path.unlink()
        except Exception as e:
            msg = f"无法删除已存在的目标文件: {target_path} - {e}"
            if logger:
                logger.warning(msg)
            return False, msg

    current_platform = platform.system()
    actual_method = link_type

    if link_type == "auto":
        if current_platform == "Windows":
            return _create_link_windows_auto(source_path, target_path, logger)
        else:
            return _create_symlink(source_path, target_path, logger)
    elif link_type == "symlink":
        return _create_symlink(source_path, target_path, logger)
    elif link_type == "hardlink":
        return _create_hardlink(source_path, target_path, logger)
    elif link_type == "copy":
        return _create_copy(source_path, target_path, logger)
    else:
        msg = f"未知的链接类型: {link_type}"
        if logger:
            logger.error(msg)
        return False, msg


def _create_symlink(
    source_path: Path,
    target_path: Path,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """
    创建符号链接

    Args:
        source_path: 源文件路径
        target_path: 目标链接路径
        logger: 日志记录器

    Returns:
        Tuple[bool, str]: (是否成功, 方法描述)
    """
    try:
        os.symlink(str(source_path), str(target_path))
        if logger:
            logger.info(f"创建符号链接成功: {target_path} -> {source_path}")
        return True, "符号链接"
    except OSError as e:
        msg = f"创建符号链接失败: {e}"
        if logger:
            logger.warning(msg)
        return False, msg
    except Exception as e:
        msg = f"创建符号链接异常: {e}"
        if logger:
            logger.error(msg)
        return False, msg


def _create_hardlink(
    source_path: Path,
    target_path: Path,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """
    创建硬链接

    注意: 硬链接仅能在同一文件系统内创建，且仅适用于文件（不适用于目录）

    Args:
        source_path: 源文件路径
        target_path: 目标链接路径
        logger: 日志记录器

    Returns:
        Tuple[bool, str]: (是否成功, 方法描述)
    """
    try:
        os.link(str(source_path), str(target_path))
        if logger:
            logger.info(f"创建硬链接成功: {target_path} <-> {source_path}")
        return True, "硬链接"
    except OSError as e:
        msg = f"创建硬链接失败: {e}"
        if logger:
            logger.warning(msg)
        return False, msg
    except Exception as e:
        msg = f"创建硬链接异常: {e}"
        if logger:
            logger.error(msg)
        return False, msg


def _create_copy(
    source_path: Path,
    target_path: Path,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """
    复制文件

    Args:
        source_path: 源文件路径
        target_path: 目标路径
        logger: 日志记录器

    Returns:
        Tuple[bool, str]: (是否成功, 方法描述)
    """
    try:
        shutil.copy2(source_path, target_path)
        if logger:
            logger.info(f"文件复制成功: {target_path} <- {source_path}")
        return True, "文件复制"
    except Exception as e:
        msg = f"文件复制失败: {e}"
        if logger:
            logger.error(msg)
        return False, msg


def _create_link_windows_auto(
    source_path: Path,
    target_path: Path,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """
    Windows平台自动选择最佳链接方式

    策略顺序：
    1. 符号链接（可能需要管理员权限）
    2. 硬链接（同一文件系统）
    3. 复制（最终回退）

    Args:
        source_path: 源文件路径
        target_path: 目标路径
        logger: 日志记录器

    Returns:
        Tuple[bool, str]: (是否成功, 实际使用的方法)
    """
    success, result = _create_symlink(source_path, target_path, logger)
    if success:
        return True, "符号链接"

    if logger:
        logger.info("符号链接失败，尝试硬链接...")

    success, result = _create_hardlink(source_path, target_path, logger)
    if success:
        return True, "硬链接"

    if logger:
        logger.info("硬链接失败，回退到文件复制...")

    success, result = _create_copy(source_path, target_path, logger)
    if success:
        return True, "文件复制(回退)"

    return False, result


def is_link_file(file_path: Path) -> bool:
    """
    判断文件是否为链接文件（符号链接或硬链接）

    Args:
        file_path: 文件路径

    Returns:
        bool: 是否为链接文件
    """
    try:
        if file_path.is_symlink():
            return True
        if hasattr(os, "path") and hasattr(os.path, "samefile"):
            stat = file_path.stat()
            if stat.st_nlink > 1:
                return True
    except Exception:
        pass
    return False


def get_link_target(file_path: Path) -> Optional[Path]:
    """
    获取链接文件的目标路径

    Args:
        file_path: 链接文件路径

    Returns:
        Optional[Path]: 目标路径，如果不是链接则返回None
    """
    try:
        if file_path.is_symlink():
            return Path(os.readlink(str(file_path)))
    except Exception:
        pass
    return None