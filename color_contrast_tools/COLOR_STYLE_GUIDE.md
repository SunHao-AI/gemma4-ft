# Notebook UI 颜色方案样式指南
## WCAG 2.1 AA 无障碍标准实施

### 概述

本样式指南定义了 Gemma4 Multimodal Demo 项目中 Jupyter Notebook UI 的颜色方案,确保所有文字和背景颜色组合符合 WCAG 2.1 AA 无障碍标准。

### WCAG 2.1 AA 对比度要求

根据 WCAG (Web Content Accessibility Guidelines) 2.1 AA 标准:

| 文字类型 | 最小对比度比率 | 示例 |
|---------|--------------|------|
| 普通文字 | 4.5:1 | 小于 18px 的文字 |
| 大文字 | 3:0:1 | 18px 或更大,或 14px 加粗 |
| 图形用户界面组件 | 3.0:1 | 按钮、图标、边框 |

**计算公式**: `(L1 + 0.05) / (L2 + 0.05)`,其中 L1 是较亮颜色的相对亮度,L2 是较暗颜色的相对亮度。

### 标准颜色方案

#### 1. 步骤标题卡片 (Step Header)

**使用场景**: 显示处理流程的各个步骤

| 元素 | 背景色 | 文字色 | 对比度 | 符合标准 |
|------|--------|--------|--------|----------|
| 渐变背景开始 | `#667eea` (蓝紫色) | `#ffffff` (白色) | 3.66:1 | ✓ AA (大文字) |
| 渐变背景结束 | `#764ba2` (紫色) | `#ffffff` (白色) | 6.37:1 | ✓ AA (大文字) |
| 已完成步骤标记 | `#667eea` | `#000000` (黑色) | 5.74:1 | ✓ AA (普通文字) |
| 当前步骤标记 | `#667eea` | `#ffffff` (白色) | 3.66:1 | ✓ AA (大文字) |
| 进度条背景 | `#00ff88` (亮绿色) | `#333` (深灰) | 11.37:1 | ✓ AA (普通文字) |

**CSS 示例**:
```css
.step-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 16px;
    border-radius: 10px;
    color: white;
    margin: 10px 0;
}

.completed-marker {
    color: #000000;  /* 黑色文字,符合 WCAG AA */
    font-weight: bold;
}
```

#### 2. 成功卡片 (Success Card)

**使用场景**: 显示成功消息、完成状态

| 元素 | 背景色 | 文字色 | 对比度 | 符合标准 |
|------|--------|--------|--------|----------|
| 卡片背景 | `#d4edda` (浅绿) | - | - | - |
| 主标题文字 | `#d4edda` | `#155724` (深绿) | 6.99:1 | ✓ AA, ✓ AAA |
| 次要文字 | `#d4edda` | `#2d5a2d` (深绿) | 6.48:1 | ✓ AA, ✓ AAA |
| 边框 | `#c3e6cb` | - | - | - |

**CSS 示例**:
```css
.success-card {
    background: #d4edda;
    border: 2px solid #c3e6cb;
    padding: 16px;
    border-radius: 10px;
    margin: 10px 0;
}

.success-title {
    color: #155724;  /* 主标题 */
    font-size: 16px;
    font-weight: bold;
}

.success-secondary {
    color: #2d5a2d;  /* 次要文字,符合 WCAG AA */
    font-size: 12px;
}
```

#### 3. 错误卡片 (Error Card)

**使用场景**: 显示错误消息、失败状态

| 元素 | 背景色 | 文字色 | 对比度 | 符合标准 |
|------|--------|--------|--------|----------|
| 卡片背景 | `#f8d7da` (浅红) | - | - | - |
| 主标题文字 | `#f8d7da` | `#721c24` (深红) | 8.25:1 | ✓ AA, ✓ AAA |
| 次要文字 | `#f8d7da` | `#5a2d28` (深红褐) | 8.54:1 | ✓ AA, ✓ AAA |
| 边框 | `#f5c6cb` | - | - | - |

**CSS 示例**:
```css
.error-card {
    background: #f8d7da;
    border: 2px solid #f5c6cb;
    padding: 16px;
    border-radius: 10px;
    margin: 10px 0;
}

.error-title {
    color: #721c24;
    font-size: 14px;
    font-weight: bold;
}

.error-secondary {
    color: #5a2d28;  /* 次要提示文字,符合 WCAG AA */
    font-size: 12px;
}
```

#### 4. 警告卡片 (Warning Card)

**使用场景**: 显示警告消息、注意事项

| 元素 | 背景色 | 文字色 | 对比度 | 符合标准 |
|------|--------|--------|--------|----------|
| 卡片背景 | `#fff3cd` (浅黄) | - | - | - |
| 文字 | `#fff3cd` | `#5a2d28` (深红褐) | 10.30:1 | ✓ AA, ✓ AAA |
| 边框 | `#ffeeba` | - | - | - |

**CSS 示例**:
```css
.warning-card {
    background: #fff3cd;
    border: 2px solid #ffeeba;
    padding: 12px;
    border-radius: 8px;
    margin: 8px 0;
}

.warning-text {
    color: #5a2d28;  /* 符合 WCAG AA */
}
```

#### 5. 信息卡片 (Info Card)

**使用场景**: 显示信息提示、说明

| 元素 | 背景色 | 文字色 | 对比度 | 符合标准 |
|------|--------|--------|--------|----------|
| 卡片背景 | `#e7f3ff` (浅蓝) | - | - | - |
| 文字 | `#e7f3ff` | `#004085` (深蓝) | 9.01:1 | ✓ AA, ✓ AAA |
| 边框 | `#b8daff` | - | - | - |

**CSS 示例**:
```css
.info-card {
    background: #e7f3ff;
    border: 1px solid #b8daff;
    padding: 12px;
    border-radius: 8px;
    margin: 8px 0;
}

.info-text {
    color: #004085;  /* 符合 WCAG AA */
}
```

#### 6. 配置卡片 (Config Card)

**使用场景**: 显示配置参数、设置

| 元素 | 背景色 | 文字色 | 对比度 | 符合标准 |
|------|--------|--------|--------|----------|
| 卡片背景 | `#f8f9fa` (浅灰) | - | - | - |
| 标题文字 | `#f8f9fa` | `#495057` (深灰) | 7.76:1 | ✓ AA, ✓ AAA |
| 边框 | `#dee2e6` | - | - | - |

**CSS 示例**:
```css
.config-card {
    background: #f8f9fa;
    padding: 16px;
    border-radius: 10px;
    margin: 10px 0;
    border: 1px solid #dee2e6;
}

.config-title {
    color: #495057;  /* 符合 WCAG AA */
    font-weight: bold;
}
```

#### 7. 最终总结卡片 (Final Summary)

**使用场景**: 显示流程完成总结

| 元素 | 背景色 | 文字色 | 对比度 | 符合标准 |
|------|--------|--------|--------|----------|
| 渐变背景开始 | `#28a745` (绿色) | `#ffffff` (白色) | 3.13:1 | ✓ AA (大文字) |
| 渐变背景结束 | `#1a8060` (青绿) | `#ffffff` (白色) | 4.88:1 | ✓ AA (大文字) |
| 内部表格背景 | `#ffffff` (白色) | `#333` (深灰) | 12.63:1 | ✓ AA, ✓ AAA |

**CSS 示例**:
```css
.final-summary {
    background: linear-gradient(135deg, #28a745, #1a8060);
    padding: 20px;
    border-radius: 12px;
    color: white;
    margin: 10px 0;
}

.summary-inner-table {
    background: white;
    color: #333;  /* 高对比度 */
    border-radius: 8px;
    padding: 12px;
}
```

### 颜色改进历史

以下颜色经过改进以达到 WCAG 2.1 AA 标准:

| 元素 | 原始颜色 | 改进后颜色 | 原对比度 | 改进后对比度 | 改进原因 |
|------|---------|-----------|----------|-------------|----------|
| 成功卡片次要文字 | `#6c757d` | `#2d5a2d` | 3.78:1 ❌ | 6.48:1 ✓ | 加深绿色以提高对比度 |
| 错误卡片次要文字 | `#856404` | `#5a2d28` | 4.11:1 ❌ | 8.54:1 ✓ | 使用深红褐替代棕色 |
| 警告卡片文字 | `#856404` | `#5a2d28` | 4.96:1 ✓ | 10.30:1 ✓ | 提高对比度至 AAA 级别 |
| 已完成步骤标记 | `#00ff88` | `#000000` | 2.73:1 ❌ | 5.74:1 ✓ | 使用黑色替代亮绿色 |
| 最终总结渐变结束 | `#20c997` | `#1a8060` | 2.13:1 ❌ | 4.88:1 ✓ | 深化背景色以提高对比度 |

### 设计原则

1. **视觉语义一致性**: 
   - 成功消息使用绿色系列
   - 错误消息使用红色系列
   - 警告消息使用黄色系列
   - 信息提示使用蓝色系列

2. **对比度优先**: 
   - 所有普通文字至少 4.5:1 对比度
   - 所有大文字至少 3.0:1 对比度
   - 优先达到 AAA 级别 (7:1 或 4.5:1)

3. **色彩层次**: 
   - 主要信息使用更深的文字颜色
   - 次要信息使用适中的文字颜色
   - 保持视觉层次分明

### 测试方法

#### 使用在线工具测试

1. **WebAIM Contrast Checker**: https://webaim.org/resources/contrastchecker/
2. **Colour Contrast Analyser (CCA)**: https://www.tpgi.com/color-contrast-checker/
3. **Chrome DevTools**: 
   - 打开开发者工具
   - 选择元素
   - 在 "Accessibility" 标签中查看对比度

#### 使用项目内置工具

项目提供了 Python 工具进行对比度计算:

```bash
# 验证所有颜色方案
python color_contrast_validation.py

# 分析现有颜色
python color_contrast_analyzer_v2.py

# 生成改进建议
python color_contrast_final.py
```

### 实施建议

1. **新增颜色时**: 
   - 使用对比度计算工具验证
   - 确保 >= 4.5:1 (普通文字) 或 >= 3:1 (大文字)
   - 记录在样式指南中

2. **修改颜色时**: 
   - 先运行对比度分析工具
   - 确认修改后符合标准
   - 更新样式指南文档

3. **设计新卡片时**: 
   - 参考现有卡片颜色方案
   - 保持视觉语义一致
   - 验证对比度符合标准

### 常见问题

**Q: 为什么大文字只需要 3:1 对比度?**
A: 大文字 (>=18px 或 >=14px 加粗) 更容易阅读,因此 WCAG 标准允许较低的对比度。

**Q: 如何判断文字是大文字还是普通文字?**
A: 
- 大文字: >=18px,或 >=14px 且加粗
- 普通文字: <18px,或 <14px 加粗

**Q: 渐变背景如何测试对比度?**
A: 测试渐变的开始和结束颜色分别与文字颜色的对比度,确保两端都符合标准。

**Q: 半透明颜色如何测试?**
A: WCAG 不直接定义半透明颜色的对比度计算。建议:
- 测试叠加在最可能背景上的最终颜色
- 确保在各种背景上都保持足够的对比度

### 参考资源

- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/#contrast-minimum)
- [Understanding WCAG Contrast Requirements](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html)
- [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)
- [Colour Contrast Analyser Tool](https://www.tpgi.com/color-contrast-checker/)

---

**最后更新**: 2026-05-08  
**版本**: 1.0  
**维护者**: AI Color Contrast Analysis System