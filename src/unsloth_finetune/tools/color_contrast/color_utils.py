#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WCAG 2.1 颜色对比度计算工具模块

提供颜色转换和对比度计算的核心功能，包括：
- 十六进制颜色与RGB值的相互转换
- sRGB到线性RGB的转换
- 相对亮度计算
- 对比度比率计算
- WCAG合规性判断

Usage:
    from color_utils import hex_to_rgb, calculate_contrast_ratio, wcag_compliance

    # 计算两个颜色的对比度
    ratio = calculate_contrast_ratio('#ffffff', '#000000')
    print(f"对比度: {ratio:.2f}:1")  # 21.00:1

    # 检查WCAG合规性
    compliance = wcag_compliance(ratio)
    print(f"AA标准: {'通过' if compliance['AA_normal'] else '未通过'}")
"""

from typing import Tuple


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    将十六进制颜色转换为 RGB 值

    支持 #RGB 和 #RRGGBB 两种格式，# 前缀可选。

    Args:
        hex_color: 十六进制颜色字符串，例如 '#ff0000' 或 '#f00'

    Returns:
        包含三个整数的元组 (R, G, B)，每个值的范围是 0-255

    Examples:
        >>> hex_to_rgb('#ff0000')
        (255, 0, 0)
        >>> hex_to_rgb('#f00')
        (255, 0, 0)
        >>> hex_to_rgb('00ff00')
        (0, 255, 0)
    """
    hex_color = hex_color.strip().lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    将 RGB 值转换为十六进制颜色

    Args:
        r: 红色通道值 (0-255)
        g: 绿色通道值 (0-255)
        b: 蓝色通道值 (0-255)

    Returns:
        十六进制颜色字符串，格式为 #RRGGBB

    Examples:
        >>> rgb_to_hex(255, 0, 0)
        '#ff0000'
        >>> rgb_to_hex(0, 128, 255)
        '#0080ff'
    """
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def srgb_to_linear(channel: int) -> float:
    """
    将 sRGB 颜色通道转换为线性亮度值

    根据 IEC 61966-2-1 标准，sRGB 到线性 RGB 的转换公式：
    - 当 c <= 0.03928 时，线性值 = c / 12.92
    - 当 c > 0.03928 时，线性值 = ((c + 0.055) / 1.055) ^ 2.4

    Args:
        channel: sRGB 颜色通道值 (0-255)

    Returns:
        线性亮度值 (0.0-1.0)

    Examples:
        >>> srgb_to_linear(0)
        0.0
        >>> srgb_to_linear(255)
        1.0
    """
    c = channel / 255.0
    if c <= 0.03928:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4


def get_relative_luminance(r: int, g: int, b: int) -> float:
    """
    计算相对亮度值 (0-1)

    根据 WCAG 2.1 标准，相对亮度计算公式：
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B

    其中 R、G、B 是线性 RGB 值（需要先通过 srgb_to_linear 转换）

    Args:
        r: 红色通道值 (0-255)
        g: 绿色通道值 (0-255)
        b: 蓝色通道值 (0-255)

    Returns:
        相对亮度值 (0.0-1.0)

    Examples:
        >>> get_relative_luminance(0, 0, 0)  # 黑色
        0.0
        >>> get_relative_luminance(255, 255, 255)  # 白色
        1.0
    """
    r_linear = srgb_to_linear(r)
    g_linear = srgb_to_linear(g)
    b_linear = srgb_to_linear(b)
    return 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear


def calculate_contrast_ratio(color1: str, color2: str) -> float:
    """
    计算两个颜色之间的对比度比率

    根据 WCAG 2.1 标准，对比度比率计算公式：
    (L1 + 0.05) / (L2 + 0.05)

    其中 L1 是较亮颜色的相对亮度，L2 是较暗颜色的相对亮度

    Args:
        color1: 第一个颜色的十六进制值，例如 '#ffffff'
        color2: 第二个颜色的十六进制值，例如 '#000000'

    Returns:
        对比度比率，范围是 1:1 到 21:1

    Examples:
        >>> calculate_contrast_ratio('#ffffff', '#000000')  # 黑白对比度
        21.0
        >>> calculate_contrast_ratio('#ff0000', '#00ff00')
        2.91...
    """
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)

    l1 = get_relative_luminance(*rgb1)
    l2 = get_relative_luminance(*rgb2)

    lighter = max(l1, l2)
    darker = min(l1, l2)

    return (lighter + 0.05) / (darker + 0.05)


def wcag_compliance(contrast_ratio: float) -> dict:
    """
    判断对比度是否符合 WCAG 2.1 AA 和 AAA 标准

    WCAG 2.1 对比度要求：
    - AA 级别：普通文字 >= 4.5:1，大文字 >= 3:1
    - AAA 级别：普通文字 >= 7:1，大文字 >= 4.5:1

    Args:
        contrast_ratio: 对比度比率

    Returns:
        包含合规性信息的字典，键包括：
        - 'AA_normal': bool, 普通文字是否符合 AA 标准
        - 'AA_large': bool, 大文字是否符合 AA 标准
        - 'AAA_normal': bool, 普通文字是否符合 AAA 标准
        - 'AAA_large': bool, 大文字是否符合 AAA 标准
        - 'ratio': float, 对比度比率

    Examples:
        >>> wcag_compliance(4.5)
        {'AA_normal': True, 'AA_large': True, 'AAA_normal': False, 'AAA_large': True, 'ratio': 4.5}
        >>> wcag_compliance(2.0)
        {'AA_normal': False, 'AA_large': False, 'AAA_normal': False, 'AAA_large': False, 'ratio': 2.0}
    """
    return {
        'AA_normal': contrast_ratio >= 4.5,
        'AA_large': contrast_ratio >= 3.0,
        'AAA_normal': contrast_ratio >= 7.0,
        'AAA_large': contrast_ratio >= 4.5,
        'ratio': contrast_ratio
    }
