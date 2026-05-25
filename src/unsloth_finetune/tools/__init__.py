"""Ancillary tool packages."""

from .font_utils import (
    detect_os,
    detect_package_manager,
    download_and_register_chinese_font,
    find_and_register_chinese_font,
    get_cached_font_path,
    get_chinese_font,
    get_font_cache_dir,
    get_system_font_dirs,
    install_chinese_font,
    refresh_matplotlib_font_cache,
    register_font_to_matplotlib,
    scan_chinese_font_files,
    setup_chinese_font,
)

__all__ = [
    "detect_os",
    "detect_package_manager",
    "download_and_register_chinese_font",
    "find_and_register_chinese_font",
    "get_cached_font_path",
    "get_chinese_font",
    "get_font_cache_dir",
    "get_system_font_dirs",
    "install_chinese_font",
    "refresh_matplotlib_font_cache",
    "register_font_to_matplotlib",
    "scan_chinese_font_files",
    "setup_chinese_font",
]
