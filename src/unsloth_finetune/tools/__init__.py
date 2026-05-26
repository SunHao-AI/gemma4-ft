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

from .labelme_to_training_format import (
    batch_labelme_to_gemma4,
    build_gemma4_detection_prompt_en,
    build_gemma4_detection_prompt_zh,
    build_gemma4_training_message,
    gemma4_to_pixel_coords,
    labelme_to_gemma4_format,
    labelme_to_gemma4_format_from_dict,
    validate_gemma4_detections,
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
    "batch_labelme_to_gemma4",
    "build_gemma4_detection_prompt_en",
    "build_gemma4_detection_prompt_zh",
    "build_gemma4_training_message",
    "gemma4_to_pixel_coords",
    "labelme_to_gemma4_format",
    "labelme_to_gemma4_format_from_dict",
    "validate_gemma4_detections",
]
