"""
LabelMe 格式数据转换为可训练格式的工具函数集合

提供多种目标模型格式的转换函数:
- Gemma4 格式: box_2d 格式,坐标归一化到 [0, 1000]
- 其他格式可扩展

每个函数独立可用,适合快速集成到数据预处理流程中。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def labelme_to_gemma4_format(
    labelme_json_path: Union[str, Path],
    image_w: int,
    image_h: int,
) -> List[Dict[str, Any]]:
    """将 LabelMe JSON 文件转换为 Gemma4 训练格式。

    Gemma4 格式特点:
    - box_2d 字段包含 [y1, x1, y2, x2] 顺序的坐标
    - 坐标归一化到 [0, 1000] 范围
    - 仅处理 rectangle 类型的标注

    Args:
        labelme_json_path: LabelMe JSON 文件路径
        image_w: 图像宽度 (像素)
        image_h: 图像高度 (像素)

    Returns:
        检测结果列表,每个元素包含:
        - box_2d: [y1, x1, y2, x2] 归一化坐标
        - label: 目标类别标签

    Example:
        >>> detections = labelme_to_gemma4_format("data/label.json", 1920, 1080)
        >>> detections
        [{'box_2d': [100, 200, 300, 400], 'label': 'cat'}]
    """
    json_path = Path(labelme_json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"LabelMe JSON 文件不存在: {labelme_json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    detections: List[Dict[str, Any]] = []

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue

        points = shape.get("points", [])
        if len(points) < 2:
            continue

        (xmin, ymin), (xmax, ymax) = points[0], points[1]
        label = shape.get("label", "object")

        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        y1 = round(ymin / image_h * 1000)
        x1 = round(xmin / image_w * 1000)
        y2 = round(ymax / image_h * 1000)
        x2 = round(xmax / image_w * 1000)

        y1 = max(0, min(1000, y1))
        x1 = max(0, min(1000, x1))
        y2 = max(0, min(1000, y2))
        x2 = max(0, min(1000, x2))

        detections.append({"box_2d": [y1, x1, y2, x2], "label": label})

    return detections


def labelme_to_gemma4_format_from_dict(
    labelme_data: Dict[str, Any],
    image_w: int,
    image_h: int,
) -> List[Dict[str, Any]]:
    """将已加载的 LabelMe 数据字典转换为 Gemma4 格式。

    Args:
        labelme_data: 已解析的 LabelMe JSON 数据字典
        image_w: 图像宽度 (像素)
        image_h: 图像高度 (像素)

    Returns:
        检测结果列表,格式同 labelme_to_gemma4_format
    """
    detections: List[Dict[str, Any]] = []

    for shape in labelme_data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue

        points = shape.get("points", [])
        if len(points) < 2:
            continue

        (xmin, ymin), (xmax, ymax) = points[0], points[1]
        label = shape.get("label", "object")

        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        y1 = round(ymin / image_h * 1000)
        x1 = round(xmin / image_w * 1000)
        y2 = round(ymax / image_h * 1000)
        x2 = round(xmax / image_w * 1000)

        y1 = max(0, min(1000, y1))
        x1 = max(0, min(1000, x1))
        y2 = max(0, min(1000, y2))
        x2 = max(0, min(1000, x2))

        detections.append({"box_2d": [y1, x1, y2, x2], "label": label})

    return detections


def batch_labelme_to_gemma4(
    labelme_json_dir: Union[str, Path],
    output_dir: Union[str, Path],
    image_size_map: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """批量转换 LabelMe JSON 文件到 Gemma4 格式。

    Args:
        labelme_json_dir: LabelMe JSON 文件目录
        output_dir: 输出目录
        image_size_map: 可选的图像尺寸映射 {filename: (w, h)}
            如果不提供,将从 JSON 文件中的 imageDataWidth/imageDataHeight 读取

    Returns:
        转换统计信息字典,包含:
        - total_files: 总文件数
        - converted: 成功转换数
        - failed: 失败数
        - output_files: 输出文件路径列表
    """
    json_dir = Path(labelme_json_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(json_dir.glob("*.json"))

    stats = {
        "total_files": len(json_files),
        "converted": 0,
        "failed": 0,
        "output_files": [],
    }

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if image_size_map and json_file.name in image_size_map:
                image_w, image_h = image_size_map[json_file.name]
            else:
                image_w = data.get("imageWidth", 1024)
                image_h = data.get("imageHeight", 1024)

            detections = labelme_to_gemma4_format_from_dict(data, image_w, image_h)

            output_file = out_dir / f"{json_file.stem}_gemma4.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(detections, f, ensure_ascii=False, indent=2)

            stats["converted"] += 1
            stats["output_files"].append(str(output_file))

        except Exception as e:
            stats["failed"] += 1

    return stats


def gemma4_to_pixel_coords(
    box_2d: List[int],
    image_w: int,
    image_h: int,
) -> Tuple[int, int, int, int]:
    """将 Gemma4 格式坐标转换为像素坐标。

    Args:
        box_2d: [y1, x1, y2, x2] 归一化坐标 (范围 [0, 1000])
        image_w: 图像宽度
        image_h: 图像高度

    Returns:
        (x_min, y_min, x_max, y_max) 像素坐标
    """
    y1, x1, y2, x2 = box_2d

    x_min = round(x1 / 1000 * image_w)
    y_min = round(y1 / 1000 * image_h)
    x_max = round(x2 / 1000 * image_w)
    y_max = round(y2 / 1000 * image_h)

    return (x_min, y_min, x_max, y_max)


def validate_gemma4_detections(
    detections: List[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """验证 Gemma4 格式检测结果的有效性。

    Args:
        detections: 检测结果列表

    Returns:
        (is_valid, error_messages):
        - is_valid: 是否全部有效
        - error_messages: 错误信息列表
    """
    errors: List[str] = []

    if not isinstance(detections, list):
        return (False, ["检测结果必须是一个列表"])

    for i, det in enumerate(detections):
        if not isinstance(det, dict):
            errors.append(f"第 {i} 个检测结果不是字典类型")
            continue

        if "box_2d" not in det:
            errors.append(f"第 {i} 个检测结果缺少 'box_2d' 字段")
            continue

        box_2d = det["box_2d"]
        if not isinstance(box_2d, list) or len(box_2d) != 4:
            errors.append(f"第 {i} 个检测结果的 'box_2d' 格式错误,应为长度为4的列表")
            continue

        for j, val in enumerate(box_2d):
            if not isinstance(val, (int, float)):
                errors.append(f"第 {i} 个检测结果的 'box_2d' 第 {j} 个值不是数字")
            elif val < 0 or val > 1000:
                errors.append(f"第 {i} 个检测结果的 'box_2d' 第 {j} 个值超出 [0, 1000] 范围")

        if "label" not in det:
            errors.append(f"第 {i} 个检测结果缺少 'label' 字段")

    return (len(errors) == 0, errors)


def build_gemma4_training_message(
    prompt_text: str,
    detections: List[Dict[str, Any]],
    image_path: str,
) -> Dict[str, Any]:
    """构建 Gemma4 训练格式的单条消息。

    Args:
        prompt_text: 用户提示文本
        detections: Gemma4 格式检测结果
        image_path: 图像路径

    Returns:
        OpenAI messages 格式的训练数据:
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": json_response}]}
            ],
            "images": [image_path]
        }
    """
    response_text = json.dumps(detections, ensure_ascii=False)

    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": response_text}]},
        ],
        "images": [image_path],
    }


def build_gemma4_detection_prompt_zh(query: str) -> str:
    """构建中文检测提示 (Gemma4 风格)。

    Args:
        query: 检测目标描述

    Returns:
        格式化的提示文本
    """
    return f"请检测图片中的{query.strip()}, 并返回边界框坐标。"


def build_gemma4_detection_prompt_en(query: str) -> str:
    """构建英文检测提示 (Gemma4 风格)。

    Args:
        query: 检测目标描述

    Returns:
        格式化的提示文本
    """
    return f"Please detect all [{query.strip()}] in the image and return their bounding boxes."