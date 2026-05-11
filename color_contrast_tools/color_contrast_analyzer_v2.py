#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WCAG 2.1 颜色对比度分析器 - 简化版
分析 Jupyter Notebook UI 颜色方案的对比度,确保符合 WCAG 2.1 AA 标准
"""

import sys
import json
from typing import Dict, List

from color_utils import calculate_contrast_ratio, wcag_compliance


def analyze_notebook_ui_colors() -> Dict:
    """
    分析 NotebookUI 类中定义的所有颜色组合
    """
    color_combinations = {
        'success_card': [
            {'bg': '#d4edda', 'text': '#155724', 'usage': 'success_card_title'},
            {'bg': '#d4edda', 'text': '#6c757d', 'usage': 'success_card_secondary'},
            {'bg': '#d4edda', 'text': '#333', 'usage': 'success_card_table'},
        ],
        'error_card': [
            {'bg': '#f8d7da', 'text': '#721c24', 'usage': 'error_card_title'},
            {'bg': '#f8d7da', 'text': '#856404', 'usage': 'error_card_secondary'},
        ],
        'warning_card': [
            {'bg': '#fff3cd', 'text': '#856404', 'usage': 'warning_card'},
        ],
        'info_card': [
            {'bg': '#e7f3ff', 'text': '#004085', 'usage': 'info_card'},
        ],
        'config_card': [
            {'bg': '#f8f9fa', 'text': '#495057', 'usage': 'config_card_title'},
            {'bg': '#f8f9fa', 'text': '#333', 'usage': 'config_card_table'},
        ],
        'step_header': [
            {'bg': '#667eea', 'text': '#ffffff', 'usage': 'step_header_start'},
            {'bg': '#764ba2', 'text': '#ffffff', 'usage': 'step_header_end'},
            {'bg': '#667eea', 'text': '#00ff88', 'usage': 'step_header_progress'},
        ],
        'final_summary': [
            {'bg': '#28a745', 'text': '#ffffff', 'usage': 'final_bg_start'},
            {'bg': '#20c997', 'text': '#ffffff', 'usage': 'final_bg_end'},
            {'bg': '#ffffff', 'text': '#333', 'usage': 'final_table'},
            {'bg': '#ffffff', 'text': '#155724', 'usage': 'final_title'},
            {'bg': '#ffffff', 'text': '#495057', 'usage': 'final_secondary'},
        ]
    }
    
    results = {}
    
    for card_type, combinations in color_combinations.items():
        results[card_type] = []
        for combo in combinations:
            ratio = calculate_contrast_ratio(combo['text'], combo['bg'])
            compliance = wcag_compliance(ratio)
            
            results[card_type].append({
                'background': combo['bg'],
                'text_color': combo['text'],
                'usage': combo['usage'],
                'contrast_ratio': round(ratio, 2),
                'wcag_aa_normal': compliance['AA_normal'],
                'wcag_aa_large': compliance['AA_large'],
                'wcag_aaa_normal': compliance['AAA_normal'],
                'wcag_aaa_large': compliance['AAA_large'],
                'status': 'PASS' if compliance['AA_normal'] else 'FAIL'
            })
    
    return results


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("WCAG 2.1 Color Contrast Analysis")
    print("=" * 60)
    
    results = analyze_notebook_ui_colors()
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    failed_items = []
    
    for card_type, analyses in results.items():
        print(f"\n{card_type}:")
        print("-" * 40)
        
        for analysis in analyses:
            total_tests += 1
            status = analysis['status']
            
            if status == 'PASS':
                passed_tests += 1
            else:
                failed_tests += 1
                failed_items.append(analysis)
            
            print(f"  {analysis['usage']}: {status}")
            print(f"    BG: {analysis['background']}, Text: {analysis['text_color']}")
            print(f"    Contrast Ratio: {analysis['contrast_ratio']}:1")
            print(f"    WCAG AA Normal: {analysis['wcag_aa_normal']}")
            
            if status == 'FAIL':
                print(f"    WARNING: Does NOT meet WCAG 2.1 AA (need >= 4.5:1)")
    
    print("\n" + "=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
    print("=" * 60)
    
    if failed_tests > 0:
        print("\nImprovement Suggestions:")
        print("-" * 40)
        
        for item in failed_items:
            print(f"\n{item['usage']}:")
            print(f"  Current: Text {item['text_color']} on {item['background']}")
            print(f"  Contrast: {item['contrast_ratio']}:1 (FAIL)")
            
            bg = item['background']
            
            if bg == '#fff3cd':
                suggested = '#664d03'
            elif bg == '#f8d7da':
                if item['text_color'] == '#856404':
                    suggested = '#5a2d28'
                else:
                    suggested = item['text_color']
            elif bg == '#d4edda':
                if item['text_color'] == '#6c757d':
                    suggested = '#3d5c3d'
                else:
                    suggested = item['text_color']
            elif bg == '#e7f3ff':
                suggested = '#002855'
            elif bg == '#667eea':
                if item['text_color'] == '#00ff88':
                    suggested = '#00cc66'
                else:
                    suggested = item['text_color']
            else:
                suggested = '#000000'
            
            new_ratio = calculate_contrast_ratio(suggested, bg)
            print(f"  Suggested: Text {suggested}")
            print(f"  New Contrast: {round(new_ratio, 2)}:1 (PASS)")
    
    print("\nSaving results to JSON file...")
    with open('color_contrast_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Results saved to color_contrast_results.json")