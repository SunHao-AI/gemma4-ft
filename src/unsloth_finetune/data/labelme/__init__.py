"""LabelMe data processing domain package."""

from .progress_logger import (
    IN_NOTEBOOK,
    TQDM_AVAILABLE,
    create_progress_bar,
    setup_progress_logging,
    PhaseProgressManager,
)
from .file_utils import (
    ORJSON_AVAILABLE,
    json_loads,
    json_dumps_str,
    parse_json_file,
    write_json_file,
    find_json_files,
    find_image_file,
)
from .labelme_cleaner import (
    clean_labelme_data,
    CleaningResult,
)
from .labelme_converter import (
    convert_to_unsloth_format,
    ConversionResult,
    DatasetSplit,
    ConversionRecord,
    BoundingBox,
    LabelMeConverter,
)
from .detection_format import (
    OutputFormat,
    DetectionFormatSpec,
    FORMAT_SPECS,
    CoordNorm,
    CoordFormat,
    GenStrategy,
    SplitMethod,
    OutputSchema,
    PromptLang,
    PromptStyle,
    build_box_2d_json_response,
    build_detection_prompt,
    build_detection_response,
    load_prompt_template_yaml,
    build_cn_normalized_detection_prompt,
    build_en_normalized_detection_prompt,
    parse_box_2d_json_ground_truth,
    DetectionPromptBuilder,
)
from .labelme_sampler import (
    select_balanced_samples,
    BalancedSelectionResult,
    SelectionResult,
    SelectionMode,
)
from .labelme_statistics import (
    statistics_labelme_labels,
    LabelStatistics,
)

__all__ = [
    # Progress utilities
    "IN_NOTEBOOK",
    "TQDM_AVAILABLE",
    "create_progress_bar",
    "setup_progress_logging",
    "PhaseProgressManager",
    # File utilities
    "ORJSON_AVAILABLE",
    "json_loads",
    "json_dumps_str",
    "parse_json_file",
    "write_json_file",
    "find_json_files",
    "find_image_file",
    # Cleaning
    "clean_labelme_data",
    "CleaningResult",
    # Conversion
    "convert_to_unsloth_format",
    "ConversionResult",
    "DatasetSplit",
    "ConversionRecord",
    "BoundingBox",
    "LabelMeConverter",
    # Detection Format
    "OutputFormat",
    "DetectionFormatSpec",
    "FORMAT_SPECS",
    "CoordNorm",
    "CoordFormat",
    "GenStrategy",
    "SplitMethod",
    "OutputSchema",
    "PromptLang",
    "PromptStyle",
    "build_box_2d_json_response",
    "build_detection_prompt",
    "build_detection_response",
    "load_prompt_template_yaml",
    "build_cn_normalized_detection_prompt",
    "build_en_normalized_detection_prompt",
    "parse_box_2d_json_ground_truth",
    "DetectionPromptBuilder",
    # Sampling
    "select_balanced_samples",
    "BalancedSelectionResult",
    "SelectionResult",
    "SelectionMode",
    # Statistics
    "statistics_labelme_labels",
    "LabelStatistics",
]