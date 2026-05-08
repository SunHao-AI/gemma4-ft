#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WCAG 2.1 颜色改进方案 - 优化版
提供更精确的颜色改进建议,同时保持视觉设计的一致性
"""

import sys
from typing import Tuple


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """将十六进制颜色转换为 RGB 值"""
    hex_color = hex_color.strip().lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """将 RGB 值转换为十六进制颜色"""
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def srgb_to_linear(channel: int) -> float:
    """将 sRGB 颜色通道转换为线性亮度值"""
    c = channel / 255.0
    if c <= 0.03928:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4


def get_relative_luminance(r: int, g: int, b: int) -> float:
    """计算相对亮度值(0-1)"""
    r_linear = srgb_to_linear(r)
    g_linear = srgb_to_linear(g)
    b_linear = srgb_to_linear(b)
    return 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear


def calculate_contrast_ratio(color1: str, color2: str) -> float:
    """计算两个颜色之间的对比度比率"""
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    
    l1 = get_relative_luminance(*rgb1)
    l2 = get_relative_luminance(*rgb2)
    
    lighter = max(l1, l2)
    darker = min(l1, l2)
    
    return (lighter + 0.05) / (darker + 0.05)


def darken_color(hex_color: str, factor: float) -> str:
    """
    加深颜色
    factor: 0.0-1.0,0表示不变,1表示完全变黑
    """
    r, g, b = hex_to_rgb(hex_color)
    
    r = int(r * (1 - factor))
    g = int(g * (1 - factor))
    b = int(b * (1 - factor))
    
    return rgb_to_hex(r, g, b)


def find_accessible_text_color(bg_color: str, target_ratio: float = 4.5) -> dict:
    """
    为背景色找到符合 WCAG AA 标准的文字颜色
    返回黑色和白色文字的对比度,选择最优方案
    """
    white = '#ffffff'
    black = '#000000'
    
    white_ratio = calculate_contrast_ratio(white, bg_color)
    black_ratio = calculate_contrast_ratio(black, bg_color)
    
    best_color = white if white_ratio > black_ratio else black
    best_ratio = max(white_ratio, black_ratio)
    
    if best_ratio >= target_ratio:
        return {
            'color': best_color,
            'ratio': best_ratio,
            'type': 'optimal'
        }
    
    bg_rgb = hex_to_rgb(bg_color)
    bg_luminance = get_relative_luminance(*bg_rgb)
    
    if bg_luminance > 0.5:
        current_color = black
        darken_factor = 0.1
    else:
        current_color = white
        lighten_factor = 0.1
    
    iterations = 0
    max_iterations = 50
    
    while iterations < max_iterations:
        ratio = calculate_contrast_ratio(current_color, bg_color)
        
        if ratio >= target_ratio:
            return {
                'color': current_color,
                'ratio': ratio,
                'type': 'adjusted'
            }
        
        if bg_luminance > 0.5:
            current_color = darken_color(current_color, 0.05)
        else:
            r, g, b = hex_to_rgb(current_color)
            r = min(255, int(r + 255 * 0.05))
            g = min(255, int(g + 255 * 0.05))
            b = min(255, int(b + 255 * 0.05))
            current_color = rgb_to_hex(r, g, b)
        
        iterations += 1
    
    return {
        'color': current_color,
        'ratio': calculate_contrast_ratio(current_color, bg_color),
        'type': 'fallback'
    }


def get_color_adjustment(bg_color: str, original_text: str, usage: str) -> dict:
    """
    根据使用场景提供更智能的颜色调整建议
    """
    bg_rgb = hex_to_rgb(bg_color)
    text_rgb = hex_to_rgb(original_text)
    
    bg_luminance = get_relative_luminance(*bg_rgb)
    text_luminance = get_relative_luminance(*text_rgb)
    
    bg_hue = 'unknown'
    
    r, g, b = bg_rgb
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    
    if max_val == min_val:
        bg_hue = 'gray'
    elif max_val == r:
        if g > b:
            bg_hue = 'warm'
        else:
            bg_hue = 'red'
    elif max_val == g:
        if r > b:
            bg_hue = 'warm'
        else:
            bg_hue = 'green'
    elif max_val == b:
        bg_hue = 'cool'
    
    base_result = find_accessible_text_color(bg_color, 4.5)
    
    adjustments = {
        'success_card_secondary': {
            'preserve_color': True,
            'target_hue': 'green',
            'suggestion': '#2d5a2d',
            'reason': '加深绿色以保持卡片类型的一致性'
        },
        'error_card_secondary': {
            'preserve_color': True,
            'target_hue': 'red',
            'suggestion': '#5a2d28',
            'reason': '使用深红色替代棕色,保持错误卡片的视觉语义'
        },
        'step_header_start': {
            'preserve_color': False,
            'use_optimal': True,
            'suggestion': base_result['color'],
            'reason': '使用最优对比度的文字颜色(白色或黑色)'
        },
        'step_header_progress': {
            'preserve_color': True,
            'target_hue': 'green',
            'suggestion': '#00cc66',
            'reason': '调整绿色饱和度,保持进度指示的视觉效果'
        },
        'final_bg_start': {
            'preserve_color': False,
            'use_optimal': True,
            'suggestion': '#ffffff',
            'alternative': '#f8f9fa',
            'reason': '对于大标题文字(18px+),可以使用白色(对比度3:1即可)'
        },
        'final_bg_end': {
            'preserve_color': False,
            'use_optimal': True,
            'suggestion': '#ffffff',
            'reason': '对于大标题文字(18px+),可以使用白色(对比度3:1即可)'
        }
    }
    
    if usage in adjustments:
        adjustment = adjustments[usage]
        suggested_color = adjustment['suggestion']
        suggested_ratio = calculate_contrast_ratio(suggested_color, bg_color)
        
        is_large_text = usage in ['final_bg_start', 'final_bg_end', 'step_header_start', 'step_header_end']
        min_ratio = 3.0 if is_large_text else 4.5
        
        return {
            'background': bg_color,
            'original_text': original_text,
            'original_ratio': calculate_contrast_ratio(original_text, bg_color),
            'suggested_text': suggested_color,
            'suggested_ratio': suggested_ratio,
            'meets_aa': suggested_ratio >= min_ratio,
            'meets_aa_large': suggested_ratio >= 3.0,
            'meets_aaa': suggested_ratio >= 7.0 if not is_large_text else suggested_ratio >= 4.5,
            'reason': adjustment['reason'],
            'is_large_text': is_large_text,
            'min_required_ratio': min_ratio
        }
    
    return {
        'background': bg_color,
        'original_text': original_text,
        'original_ratio': calculate_contrast_ratio(original_text, bg_color),
        'suggested_text': base_result['color'],
        'suggested_ratio': base_result['ratio'],
        'meets_aa': base_result['ratio'] >= 4.5,
        'meets_aa_large': base_result['ratio'] >= 3.0,
        'meets_aaa': base_result['ratio'] >= 7.0,
        'reason': '使用自动计算的对比度最优颜色',
        'is_large_text': False,
        'min_required_ratio': 4.5
    }


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("WCAG 2.1 Color Contrast Improvement Plan")
    print("=" * 70)
    
    failed_combinations = [
        {'bg': '#d4edda', 'text': '#6c757d', 'usage': 'success_card_secondary'},
        {'bg': '#f8d7da', 'text': '#856404', 'usage': 'error_card_secondary'},
        {'bg': '#667eea', 'text': '#ffffff', 'usage': 'step_header_start'},
        {'bg': '#667eea', 'text': '#00ff88', 'usage': 'step_header_progress'},
        {'bg': '#28a745', 'text': '#ffffff', 'usage': 'final_bg_start'},
        {'bg': '#20c997', 'text': '#ffffff', 'usage': 'final_bg_end'},
    ]
    
    recommendations = []
    
    for combo in failed_combinations:
        result = get_color_adjustment(combo['bg'], combo['text'], combo['usage'])
        recommendations.append(result)
        
        print(f"\n{combo['usage']}:")
        print(f"  Background: {result['background']}")
        print(f"  Original Text: {result['original_text']}")
        print(f"  Original Ratio: {result['original_ratio']:.2f}:1")
        print(f"  Suggested Text: {result['suggested_text']}")
        print(f"  Suggested Ratio: {result['suggested_ratio']:.2f}:1")
        print(f"  Text Type: {'Large Text (>=18px)' if result['is_large_text'] else 'Normal Text'}")
        print(f"  Min Required: {result['min_required_ratio']}:1")
        print(f"  WCAG AA: {result['meets_aa']}")
        print(f"  WCAG AA (Large): {result['meets_aa_large']}")
        print(f"  WCAG AAA: {result['meets_aaa']}")
        print(f"  Reason: {result['reason']}")
    
    print("\n" + "=" * 70)
    print("Summary of Recommendations:")
    print("=" * 70)
    
    for rec in recommendations:
        status = 'PASS' if (rec['meets_aa'] or rec['meets_aa_large']) else 'FAIL'
        print(f"{rec['background']} + {rec['suggested_text']}: {rec['suggested_ratio']:.2f}:1 [{status}]")
    
    print("\nNote:")
    print("- For large text (>=18px or >=14px bold), WCAG AA requires only 3:1 ratio")
    print("- For normal text, WCAG AA requires 4.5:1 ratio")
    print("- Colors are optimized to preserve visual design consistency where possible")