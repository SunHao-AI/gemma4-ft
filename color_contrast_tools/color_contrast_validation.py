#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证修改后的颜色是否符合 WCAG 2.1 AA 标准
"""

import sys


def hex_to_rgb(hex_color: str):
    hex_color = hex_color.strip().lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def srgb_to_linear(channel: int) -> float:
    c = channel / 255.0
    if c <= 0.03928:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4


def get_relative_luminance(r: int, g: int, b: int) -> float:
    r_linear = srgb_to_linear(r)
    g_linear = srgb_to_linear(g)
    b_linear = srgb_to_linear(b)
    return 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear


def calculate_contrast_ratio(color1: str, color2: str) -> float:
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    
    l1 = get_relative_luminance(*rgb1)
    l2 = get_relative_luminance(*rgb2)
    
    lighter = max(l1, l2)
    darker = min(l1, l2)
    
    return (lighter + 0.05) / (darker + 0.05)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("WCAG 2.1 AA Validation - After Color Improvements")
    print("=" * 70)
    
    modified_colors = [
        {
            'usage': 'step_header_completed_marker',
            'bg': '#667eea',
            'original': '#00ff88',
            'modified': '#000000',
            'type': 'normal_text',
            'min_ratio': 4.5
        },
        {
            'usage': 'success_card_secondary',
            'bg': '#d4edda',
            'original': '#6c757d',
            'modified': '#2d5a2d',
            'type': 'normal_text',
            'min_ratio': 4.5
        },
        {
            'usage': 'error_card_secondary',
            'bg': '#f8d7da',
            'original': '#856404',
            'modified': '#5a2d28',
            'type': 'normal_text',
            'min_ratio': 4.5
        },
        {
            'usage': 'warning_card_text',
            'bg': '#fff3cd',
            'original': '#856404',
            'modified': '#5a2d28',
            'type': 'normal_text',
            'min_ratio': 4.5
        },
        {
            'usage': 'final_summary_bg_end',
            'bg': '#1a8060',
            'original_bg': '#20c997',
            'text': '#ffffff',
            'type': 'large_text',
            'min_ratio': 3.0
        },
    ]
    
    all_pass = True
    
    for color in modified_colors:
        print(f"\n{color['usage']} ({color['type']}):")
        
        if 'modified' in color:
            original_ratio = calculate_contrast_ratio(color['original'], color['bg'])
            modified_ratio = calculate_contrast_ratio(color['modified'], color['bg'])
            
            print(f"  Background: {color['bg']}")
            print(f"  Original Text: {color['original']} -> {original_ratio:.2f}:1")
            print(f"  Modified Text: {color['modified']} -> {modified_ratio:.2f}:1")
            
            passes = modified_ratio >= color['min_ratio']
            status = 'PASS' if passes else 'FAIL'
            print(f"  WCAG AA: {status} (Required: {color['min_ratio']}:1)")
            
            if not passes:
                all_pass = False
        
        elif 'text' in color:
            original_bg_ratio = calculate_contrast_ratio(color['text'], color['original_bg'])
            modified_bg_ratio = calculate_contrast_ratio(color['text'], color['bg'])
            
            print(f"  Text: {color['text']}")
            print(f"  Original BG: {color['original_bg']} -> {original_bg_ratio:.2f}:1")
            print(f"  Modified BG: {color['bg']} -> {modified_bg_ratio:.2f}:1")
            
            passes = modified_bg_ratio >= color['min_ratio']
            status = 'PASS' if passes else 'FAIL'
            print(f"  WCAG AA: {status} (Required: {color['min_ratio']}:1)")
            
            if not passes:
                all_pass = False
    
    print("\n" + "=" * 70)
    
    if all_pass:
        print("SUCCESS: All modified colors meet WCAG 2.1 AA standards!")
    else:
        print("WARNING: Some colors still need improvement")
    
    print("=" * 70)
    
    print("\nComplete Color Palette Summary:")
    print("-" * 70)
    
    full_palette = [
        ('step_header_bg_start', '#667eea', '#ffffff', 'large_text', 3.0),
        ('step_header_bg_end', '#764ba2', '#ffffff', 'large_text', 3.0),
        ('step_header_completed', '#667eea', '#000000', 'normal_text', 4.5),
        ('step_header_current', '#667eea', '#ffffff', 'large_text', 3.0),
        ('success_card_primary', '#d4edda', '#155724', 'normal_text', 4.5),
        ('success_card_secondary', '#d4edda', '#2d5a2d', 'normal_text', 4.5),
        ('error_card_primary', '#f8d7da', '#721c24', 'normal_text', 4.5),
        ('error_card_secondary', '#f8d7da', '#5a2d28', 'normal_text', 4.5),
        ('warning_card', '#fff3cd', '#5a2d28', 'normal_text', 4.5),
        ('info_card', '#e7f3ff', '#004085', 'normal_text', 4.5),
        ('config_card', '#f8f9fa', '#495057', 'normal_text', 4.5),
        ('final_summary_bg_start', '#28a745', '#ffffff', 'large_text', 3.0),
        ('final_summary_bg_end', '#1a8060', '#ffffff', 'large_text', 3.0),
        ('final_summary_inner', '#ffffff', '#333', 'normal_text', 4.5),
    ]
    
    for name, bg, text, text_type, min_ratio in full_palette:
        ratio = calculate_contrast_ratio(text, bg)
        status = 'PASS' if ratio >= min_ratio else 'FAIL'
        print(f"{name}: {bg} + {text} = {ratio:.2f}:1 [{status}] (need {min_ratio}:1)")
    
    print("\nWCAG 2.1 AA Standards:")
    print("- Normal text: >= 4.5:1")
    print("- Large text (>=18px or >=14px bold): >= 3:1")
    print("- AAA Normal text: >= 7:1")
    print("- AAA Large text: >= 4.5:1")