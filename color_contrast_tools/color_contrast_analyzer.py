#!/usr/bin/env python3
"""
WCAG 2.1 颜色对比度分析器
分析 Jupyter Notebook UI 颜色方案的对比度,确保符合 WCAG 2.1 AA 标准
"""

from typing import Dict, List

from color_utils import calculate_contrast_ratio, wcag_compliance


def analyze_notebook_ui_colors() -> Dict:
    """
    分析 NotebookUI 类中定义的所有颜色组合
    """
    color_combinations = {
        'success_card': [
            {'bg': '#d4edda', 'text': '#155724', 'usage': '成功卡片标题文字'},
            {'bg': '#d4edda', 'text': '#6c757d', 'usage': '成功卡片次要文字'},
            {'bg': '#d4edda', 'text': '#333', 'usage': '成功卡片表格文字'},
        ],
        'error_card': [
            {'bg': '#f8d7da', 'text': '#721c24', 'usage': '错误卡片标题文字'},
            {'bg': '#f8d7da', 'text': '#856404', 'usage': '错误卡片次要文字'},
        ],
        'warning_card': [
            {'bg': '#fff3cd', 'text': '#856404', 'usage': '警告卡片文字'},
        ],
        'info_card': [
            {'bg': '#e7f3ff', 'text': '#004085', 'usage': '信息卡片文字'},
        ],
        'config_card': [
            {'bg': '#f8f9fa', 'text': '#495057', 'usage': '配置卡片标题'},
            {'bg': '#f8f9fa', 'text': '#333', 'usage': '配置卡片表格文字'},
        ],
        'step_header': [
            {'bg': '#667eea', 'text': '#ffffff', 'usage': '步骤标题背景开始色'},
            {'bg': '#764ba2', 'text': '#ffffff', 'usage': '步骤标题背景结束色'},
            {'bg': '#667eea', 'text': '#00ff88', 'usage': '进度条绿色部分'},
        ],
        'final_summary': [
            {'bg': '#28a745', 'text': '#ffffff', 'usage': '完成卡片背景开始色'},
            {'bg': '#20c997', 'text': '#ffffff', 'usage': '完成卡片背景结束色'},
            {'bg': '#ffffff', 'text': '#333', 'usage': '完成卡片内部表格'},
            {'bg': '#ffffff', 'text': '#155724', 'usage': '完成卡片标题'},
            {'bg': '#ffffff', 'text': '#495057', 'usage': '完成卡片次要标题'},
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
                'contrast_ratio': ratio,
                'wcag_aa_normal': compliance['AA_normal'],
                'wcag_aa_large': compliance['AA_large'],
                'wcag_aaa_normal': compliance['AAA_normal'],
                'wcag_aaa_large': compliance['AAA_large'],
                'status': 'PASS' if compliance['AA_normal'] else 'FAIL'
            })
    
    return results


def generate_report(results: Dict) -> str:
    """生成详细的分析报告"""
    report = []
    report.append("=" * 80)
    report.append("WCAG 2.1 颜色对比度分析报告")
    report.append("=" * 80)
    report.append("\n")
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for card_type, analyses in results.items():
        report.append(f"\n{card_type.upper()} 卡片颜色分析:")
        report.append("-" * 60)
        
        for analysis in analyses:
            total_tests += 1
            if analysis['wcag_aa_normal']:
                passed_tests += 1
                status_icon = "✅"
            else:
                failed_tests += 1
                status_icon = "❌"
            
            report.append(f"\n{status_icon} {analysis['usage']}")
            report.append(f"   背景: {analysis['background']}")
            report.append(f"   文字: {analysis['text_color']}")
            report.append(f"   对比度: {analysis['contrast_ratio']:.2f}:1")
            report.append(f"   WCAG AA (普通文字): {analysis['wcag_aa_normal']}")
            report.append(f"   WCAG AA (大文字): {analysis['wcag_aa_large']}")
            
            if not analysis['wcag_aa_normal']:
                report.append(f"   ⚠️  不符合 WCAG 2.1 AA 标准 (需要 ≥ 4.5:1)")
    
    report.append("\n" + "=" * 80)
    report.append(f"总测试数: {total_tests}")
    report.append(f"符合标准: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    report.append(f"不符合标准: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
    report.append("=" * 80)
    
    return "\n".join(report)


def suggest_improved_colors(failed_combinations: List[Dict]) -> Dict:
    """
    为不符合标准的颜色组合提供改进建议
    """
    suggestions = {}
    
    for combo in failed_combinations:
        bg = combo['background']
        usage = combo['usage']
        
        if bg == '#fff3cd':
            suggestions[usage] = {
                'original_text': combo['text_color'],
                'suggested_text': '#664d03',
                'reason': '加深棕色以提高对比度至 4.5:1 以上'
            }
        elif bg == '#f8d7da':
            if combo['text_color'] == '#856404':
                suggestions[usage] = {
                    'original_text': combo['text_color'],
                    'suggested_text': '#5a2d28',
                    'reason': '使用更深的红色替代棕色,提高对比度'
                }
        elif bg == '#d4edda':
            if combo['text_color'] == '#6c757d':
                suggestions[usage] = {
                    'original_text': combo['text_color'],
                    'suggested_text': '#3d5c3d',
                    'reason': '加深灰色以提高对比度'
                }
        elif bg == '#e7f3ff':
            suggestions[usage] = {
                'original_text': combo['text_color'],
                'suggested_text': '#002855',
                'reason': '使用更深的蓝色以提高对比度'
            }
    
    return suggestions


if __name__ == '__main__':
    print("开始分析 NotebookUI 颜色对比度...")
    print()
    
    results = analyze_notebook_ui_colors()
    report = generate_report(results)
    print(report)
    
    print("\n" + "=" * 80)
    print("改进建议:")
    print("=" * 80)
    
    failed_combinations = []
    for card_type, analyses in results.items():
        for analysis in analyses:
            if not analysis['wcag_aa_normal']:
                failed_combinations.append(analysis)
    
    if failed_combinations:
        suggestions = suggest_improved_colors(failed_combinations)
        for usage, suggestion in suggestions.items():
            print(f"\n{usage}:")
            print(f"  原始文字颜色: {suggestion['original_text']}")
            print(f"  建议文字颜色: {suggestion['suggested_text']}")
            print(f"  原因: {suggestion['reason']}")
    else:
        print("\n[PASS] 所有颜色组合均符合 WCAG 2.1 AA 标准!")