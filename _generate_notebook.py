import json

cells = []

def md_cell(id, source_lines):
    cells.append({
        "cell_type": "markdown",
        "id": id,
        "metadata": {},
        "source": source_lines
    })

def code_cell(id, source_lines):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": id,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    })

def s(text):
    """Convert text to source array format - each line ends with \\n except last"""
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)
    return result

# Cell 1: Header
md_cell("header-intro", s("""# LabelMe标注数据处理工具

## 功能概述

本Notebook实现了LabelMe标注数据的完整处理流程，包括：

1. **标注文件清洗** — 验证JSON完整性，过滤无效数据，识别重复标注
2. **数据统计分析** — 统计类别分布、标注数量，生成可视化报告
3. **样本均衡化选择** — 支持n张图片模式和n个标签样本模式的均衡采样
4. **数据格式转换** — 转换为Unsloth框架兼容格式，生成训练/验证/测试集

## 数据处理流程

```
LabelMe JSON → [1]清洗 → [2]统计 → [3]选择 → [4]转换 → 输出数据集
```

> 所有功能已封装在 `tools` 模块，支持多线程并行处理和tqdm进度条"""))

# Cell 2: Imports + NotebookUI
code_cell("import-libraries", s("""import json
import time
import logging
import sys
from pathlib import Path

NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import (
    clean_labelme_data,
    statistics_labelme_labels,
    select_balanced_samples,
    convert_to_unsloth_format,
    SelectionMode,
    CleaningResult,
    LabelStatistics,
    BalancedSelectionResult,
    ConversionResult,
    TQDM_AVAILABLE,
)

try:
    from IPython.display import display, HTML
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    display = print
    HTML = str

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class NotebookUI:
    \"\"\"Notebook界面美化工具 — 提供统一的步骤追踪、配置展示和结果汇总\"\"\"\

    STEP_INFO = [
        ("1", "标注清洗", "验证完整性、过滤无效数据、识别重复标注"),
        ("2", "统计分析", "类别分布、标注数量统计、可视化报告"),
        ("3", "均衡选择", "基于类别的均衡采样（两种模式可选）"),
        ("4", "格式转换", "Unsloth格式转换、训练/验证/测试集划分"),
        ("5", "输出验证", "数据格式完整性验证"),
    ]

    _STYLES = {
        "step_header": "background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:16px;border-radius:10px;color:white;margin:10px 0;",
        "success_card": "background:#d4edda;border:2px solid #c3e6cb;padding:16px;border-radius:10px;margin:10px 0;",
        "error_card": "background:#f8d7da;border:2px solid #f5c6cb;padding:16px;border-radius:10px;margin:10px 0;",
        "config_card": "background:#f8f9fa;padding:16px;border-radius:10px;margin:10px 0;border:1px solid #dee2e6;",
        "info_card": "background:#e7f3ff;padding:12px;border-radius:8px;margin:8px 0;border:1px solid #b8daff;",
        "warning_card": "background:#fff3cd;border:2px solid #ffeeba;padding:12px;border-radius:8px;margin:8px 0;",
    }

    def __init__(self):
        self.completed_steps = []
        self.start_time = time.time()

    def _html(self, content):
        if IPYTHON_AVAILABLE:
            display(HTML(content))
        else:
            print(content)

    def show_step_header(self, step_num, step_name, description):
        elapsed = time.time() - self.start_time
        progress_pct = len(self.completed_steps) / len(self.STEP_INFO) * 100
        markers = ""
        for sid, sname, _ in self.STEP_INFO:
            if sid in self.completed_steps:
                markers += f"<span style='color:#00ff88;font-weight:bold'>✅{sid}</span> "
            elif sid == str(step_num):
                markers += f"<span style='color:#fff;font-weight:bold'>🔄{sid}</span> "
            else:
                markers += f"<span style='color:rgba(255,255,255,0.5)'>⏳{sid}</span> "
        self._html(f\"\"\"<div style="{self._STYLES['step_header']}">
            <div style="font-size:18px;font-weight:bold;margin-bottom:8px">🔄 步骤 {step_num}: {step_name}</div>
            <div style="font-size:13px;color:rgba(255,255,255,0.9);margin-bottom:12px">{description}</div>
            <div style="background:rgba(255,255,255,0.3);border-radius:5px;height:24px">
                <div style="background:#00ff88;border-radius:5px;height:100%;width:{progress_pct:.0f}%;display:flex;align-items:center;justify-content:center;font-weight:bold;color:#333;font-size:11px">{progress_pct:.0f}%</div>
            </div>
            <div style="margin-top:8px;font-size:12px">{markers}</div>
            <div style="font-size:12px;margin-top:6px">⏱️ 已用时 {elapsed:.1f}秒</div>
        </div>\"\"\")

    def mark_step_completed(self, step_num):
        self.completed_steps.append(str(step_num))
        elapsed = time.time() - self.start_time
        self._html(f\"\"\"<div style="{self._STYLES['success_card']}">
            <span style="font-size:16px;font-weight:bold;color:#155724">✅ 步骤 {step_num} 完成</span>
            <span style="font-size:12px;color:#6c757d;margin-left:8px">⏱️ {elapsed:.1f}秒</span>
        </div>\"\"\")

    def show_config(self, config):
        dedup = "✅ 开启" if config.deduplicate else "❌ 关闭"
        img_val = "✅ 开启" if config.validate_images else "❌ 关闭"
        rows = f\"\"\"<tr style="background:#e9ecef"><th style="padding:10px;text-align:left;border:1px solid #dee2e6">参数</th><th style="padding:10px;text-align:left;border:1px solid #dee2e6">值</th></tr>
        <tr><td style="padding:8px;border:1px solid #dee2e6;font-weight:bold">源目录</td><td style="padding:8px;border:1px solid #dee2e6">{config.source_dir}</td></tr>
        <tr><td style="padding:8px;border:1px solid #dee2e6;font-weight:bold">清洗输出</td><td style="padding:8px;border:1px solid #dee2e6">{config.cleaned_dir}</td></tr>
        <tr><td style="padding:8px;border:1px solid #dee2e6;font-weight:bold">统计输出</td><td style="padding:8px;border:1px solid #dee2e6">{config.stats_output}</td></tr>
        <tr><td style="padding:8px;border:1px solid #dee2e6;font-weight:bold">选择模式</td><td style="padding:8px;border:1px solid #dee2e6">{config.selection_mode}</td></tr>
        <tr><td style="padding:8px;border:1px solid #dee2e6;font-weight:bold">目标数量</td><td style="padding:8px;border:1px solid #dee2e6">{config.target_count}</td></tr>
        <tr><td style="padding:8px;border:1px solid #dee2e6;font-weight:bold">训练/验证/测试</td><td style="padding:8px;border:1px solid #dee2e6">{config.train_ratio}/{config.val_ratio}/{config.test_ratio}</td></tr>
        <tr><td style="padding:8px;border:1px solid #dee2e6;font-weight:bold">去重模式</td><td style="padding:8px;border:1px solid #dee2e6">{dedup}</td></tr>
        <tr><td style="padding:8px;border:1px solid #dee2e6;font-weight:bold">图片验证</td><td style="padding:8px;border:1px solid #dee2e6">{img_val}</td></tr>
        <tr><td style="padding:8px;border:1px solid #dee2e6;font-weight:bold">并行线程</td><td style="padding:8px;border:1px solid #dee2e6">{config.max_workers}</td></tr>\"\"\"
        self._html(f\"\"\"<div style="{self._STYLES['config_card']}">
            <h3 style="color:#495057;margin-bottom:12px">⚙️ 配置参数</h3>
            <table style="width:100%;border-collapse:collapse">{rows}</table>
        </div>\"\"\")

    def show_result_summary(self, title, results, success=True):
        bg = self._STYLES['success_card'] if success else self._STYLES['error_card']
        icon = "✅" if success else "⚠️"
        border_color = "#c3e6cb" if success else "#f5c6cb"
        rows = ""
        for key, value in results.items():
            rows += f"<tr><td style='padding:8px;border:1px solid {border_color};font-weight:bold'>{key}</td><td style='padding:8px;border:1px solid {border_color}'>{value}</td></tr>"
        self._html(f\"\"\"<div style="{bg}">
            <h4 style="margin-bottom:10px">{icon} {title}</h4>
            <table style="width:100%;border-collapse:collapse">{rows}</table>
        </div>\"\"\")

    def show_error(self, step_name, error_msg):
        self._html(f\"\"\"<div style="{self._STYLES['error_card']}">
            <h4 style="color:#721c24;margin-bottom:8px">❌ {step_name} 执行失败</h4>
            <div style="color:#721c24;font-size:14px">{error_msg}</div>
            <div style="color:#856404;font-size:12px;margin-top:8px">💡 请检查配置参数和数据路径后重试</div>
        </div>\"\"\")

    def show_info(self, message):
        self._html(f\"\"\"<div style="{self._STYLES['info_card']}">
            <span style="color:#004085">ℹ️ {message}</span>
        </div>\"\"\")

    def show_warning(self, message):
        self._html(f\"\"\"<div style="{self._STYLES['warning_card']}">
            <span style="color:#856404">⚠️ {message}</span>
        </div>\"\"\")

    def show_final_summary(self, cleaning_result, stats_result, selection_result, conversion_result, config):
        elapsed = time.time() - self.start_time
        items = [
            ("清洗合规文件", f"{cleaning_result.valid_count}/{cleaning_result.total_files}"),
            ("类别总数", str(stats_result.total_labels)),
            ("标注实例", str(stats_result.total_label_instances)),
            ("唯一选择图片", str(selection_result.unique_image_count)),
            ("转换记录数", str(conversion_result.converted_count)),
        ]
        if conversion_result.train_split:
            items.append(("训练集", f"{conversion_result.train_split.total_records} 条"))
        if conversion_result.val_split:
            items.append(("验证集", f"{conversion_result.val_split.total_records} 条"))
        if conversion_result.test_split:
            items.append(("测试集", f"{conversion_result.test_split.total_records} 条"))
        items.append(("总耗时", f"{elapsed:.1f} 秒"))

        rows = ""
        for key, val in items:
            rows += f"<tr><td style='padding:10px;border:1px solid #dee2e6;font-weight:bold'>{key}</td><td style='padding:10px;border:1px solid #dee2e6'>{val}</td></tr>"

        output_items = [
            ("清洗目录", config.cleaned_dir),
            ("清洗报告", cleaning_result.report_path or "N/A"),
            ("统计文件", config.stats_output),
        ]
        if conversion_result.train_split:
            output_items.append(("训练集", conversion_result.train_split.output_path))
        if conversion_result.val_split:
            output_items.append(("验证集", conversion_result.val_split.output_path))
        if conversion_result.test_split:
            output_items.append(("测试集", conversion_result.test_split.output_path))

        out_rows = ""
        for key, val in output_items:
            out_rows += f"<tr><td style='padding:10px;border:1px solid #dee2e6;font-weight:bold'>{key}</td><td style='padding:10px;border:1px solid #dee2e6;font-size:12px'>{val}</td></tr>"

        self._html(f\"\"\"<div style="background:linear-gradient(135deg,#28a745,#20c997);padding:20px;border-radius:12px;color:white;margin:10px 0">
            <h3 style="margin-bottom:15px">🎉 数据处理流程全部完成</h3>
            <div style="background:white;color:#333;border-radius:8px;padding:12px;margin-bottom:12px">
                <h4 style="margin-bottom:8px;color:#155724">处理统计</h4>
                <table style="width:100%;border-collapse:collapse">{rows}</table>
            </div>
            <div style="background:white;color:#333;border-radius:8px;padding:12px">
                <h4 style="margin-bottom:8px;color:#495057">输出文件</h4>
                <table style="width:100%;border-collapse:collapse">{out_rows}</table>
            </div>
        </div>\"\"\")


ui = NotebookUI()
ui.show_info("模块加载完成 — tqdm进度条: ✅ | matplotlib: " + ("✅" if MATPLOTLIB_AVAILABLE else "❌") + " | PIL: " + ("✅" if PIL_AVAILABLE else "❌"))"""))

# Cell 3: Config params
md_cell("config-section", s("""### 配置数据路径和处理参数

请根据实际数据路径修改 `DataProcessingConfig` 中的参数。"""))

# Cell 4: Config
code_cell("set-config", s("""class DataProcessingConfig:
    \"\"\"数据处理配置 — 请根据实际数据路径修改参数\"\"\

    def __init__(self):
        self.source_dir = r"/raid5/sh/data/wgang_40"
        self.cleaned_dir = r"/raid5/sh/data/labelme_cleaned-wgang_40"
        self.stats_output = r"/raid5/sh/data/labelme_stats-wgang_40.json"
        self.selected_dir = r"/raid5/sh/data/labelme_selected-wgang_40"
        self.output_dir = r"/raid5/sh/data/unsloth_training_data-wgang_40"
        self.log_file = r"/raid5/sh/data/processing_log-wgang_40.txt"

        self.selection_mode = "n_images"
        self.target_count = 100
        self.random_seed = 42

        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

        self.instruction_text = "请分析这张图像，识别并定位其中的目标物体。"
        self.normalize_coordinates = True
        self.validate_images = True
        self.deduplicate = True
        self.preserve_structure = True
        self.copy_images = True

        self.max_workers = 4
        self.verbose = True


config = DataProcessingConfig()
ui.show_config(config)"""))

# Cell 5: Cleaning section header (with description integrated)
md_cell("cleaning-section", s("""## 第二部分：标注文件清洗

使用 `LabelMeCleaner` 进行数据清洗，支持多线程并行验证：

- **JSON结构验证** — 检查文件格式、shapes字段完整性
- **图片关联验证** — 严格名称匹配，确保JSON与图片一一对应
- **重复标注检测** — 多个JSON指向同一图片时自动保留最优标注
- **合规文件复制** — 自动复制合规数据到目标目录，保留目录结构"""))

# Cell 6: Execute cleaning
code_cell("execute-cleaning", s("""ui.show_step_header(1, "标注清洗", "验证JSON完整性，过滤无效数据，识别重复标注")

cleaning_result = None
try:
    cleaning_result = clean_labelme_data(
        source_dir=config.source_dir,
        target_dir=config.cleaned_dir,
        preserve_structure=config.preserve_structure,
        copy_images=config.copy_images,
        deduplicate=config.deduplicate,
        generate_report=True,
        log_file=config.log_file,
        max_workers=config.max_workers,
        use_tqdm=True,
    )

    ui.mark_step_completed(1)
    ui.show_result_summary("清洗结果", {
        "总文件数": cleaning_result.total_files,
        "合规文件": f"{cleaning_result.valid_count} ({cleaning_result.valid_ratio:.1f}%)" if cleaning_result.valid_ratio else str(cleaning_result.valid_count),
        "不合规文件": cleaning_result.invalid_count,
        "重复标注": cleaning_result.duplicate_count,
        "复制JSON": len(cleaning_result.copied_json_files),
        "复制图片": len(cleaning_result.copied_image_files),
        "耗时": f"{cleaning_result.duration:.2f}秒" if cleaning_result.duration else "N/A",
    })
except Exception as e:
    ui.show_error("标注清洗", str(e))
    cleaning_result = CleaningResult()"""))

# Cell 7: Cleaning details
md_cell("cleaning-details", s("""### 查看清洗详情

显示不合规文件和重复标注的详细分类信息。"""))

# Cell 8: Show cleaning details
code_cell("show-cleaning-details", s("""if cleaning_result and cleaning_result.invalid_files:
    status_counts = {}
    for item in cleaning_result.invalid_files:
        status = item.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    rows = ""
    for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
        rows += f"<tr><td style='padding:6px;border:1px solid #f5c6cb;font-weight:bold'>{status}</td><td style='padding:6px;border:1px solid #f5c6cb'>{count} 个</td></tr>"
    ui._html(f\"\"\"<div style="{ui._STYLES['warning_card']}"><h4 style='color:#856404;margin-bottom:8px'>⚠️ 不合规文件分类</h4><table style='width:100%;border-collapse:collapse'>{rows}</table></div>\"\"\")

    print("\\n不合规文件示例（前5个）:")
    for i, item in enumerate(cleaning_result.invalid_files[:5], 1):
        print(f"  {i}. {Path(item['file']).name} — {item['reason']}")
else:
    ui.show_info("无不合规文件 ✅")

if cleaning_result and cleaning_result.duplicate_files:
    image_counts = {}
    for item in cleaning_result.duplicate_files:
        image_file = item.get("image_file", "unknown")
        image_counts[image_file] = image_counts.get(image_file, 0) + 1

    rows = ""
    for image_file, count in sorted(image_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        rows += f"<tr><td style='padding:6px;border:1px solid #ffeeba'>{Path(image_file).name}</td><td style='padding:6px;border:1px solid #ffeeba'>{count} 个重复</td></tr>"
    ui._html(f\"\"\"<div style="{ui._STYLES['warning_card']}"><h4 style='color:#856404;margin-bottom:8px'>⚠️ 重复标注文件（Top 5）</h4><table style='width:100%;border-collapse:collapse'>{rows}</table></div>\"\"\")
else:
    ui.show_info("无重复标注文件 ✅")"""))

# Cell 9: Statistics section header
md_cell("stats-section", s("""## 第三部分：数据统计分析

使用 `LabelMeLabelStatistics` 对清洗后的数据进行类别统计：

- **总文件数/处理/跳过统计** — 含无imageUrl和解析错误的分类计数
- **类别总数/标注实例** — 每个label的文件数、实例数、单文件标注范围
- **多线程并行处理** — `max_workers` 控制并行线程数
- **结构化JSON输出** — 类别按字母排序，数量键按数字排序"""))

# Cell 10: Execute statistics
code_cell("execute-stats", s("""ui.show_step_header(2, "统计分析", "类别分布、标注数量统计、可视化报告")

stats_result = None
try:
    stats_result = statistics_labelme_labels(
        source_dir=config.cleaned_dir,
        recursive=True,
        use_relative_path=True,
        max_workers=config.max_workers,
        log_file=config.log_file,
        use_tqdm=True,
    )

    stats_path = Path(config.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_result.to_structured_dict(), f, indent=2, ensure_ascii=False)

    ui.mark_step_completed(2)
    ui.show_result_summary("统计结果", {
        "总JSON文件": stats_result.total_json_files,
        "有效处理": stats_result.processed_files,
        "跳过(无imageUrl)": stats_result.skipped_no_imageurl,
        "跳过(解析错误)": stats_result.skipped_parse_error,
        "类别总数": stats_result.total_labels,
        "标注实例": stats_result.total_label_instances,
        "统计文件": config.stats_output,
    })
except Exception as e:
    ui.show_error("统计分析", str(e))
    stats_result = LabelStatistics(source_dir=config.cleaned_dir)"""))

# Cell 11: Visualization
md_cell("stats-visualize", s("""### 类别分布可视化

生成Top 15类别标注实例和文件分布的双栏对比图表。"""))

# Cell 12: Visualize
code_cell("visualize-stats", s("""if MATPLOTLIB_AVAILABLE and stats_result and stats_result.label_counts:
    summary = stats_result.get_label_summary()
    sorted_summary = sorted(summary.items(), key=lambda x: x[1]["total_instances"], reverse=True)
    top_n = min(15, len(sorted_summary))

    labels = [item[0] for item in sorted_summary[:top_n]]
    instances = [item[1]["total_instances"] for item in sorted_summary[:top_n]]
    files = [item[1]["total_files"] for item in sorted_summary[:top_n]]

    plt.rcParams.update({"font.size": 11, "figure.dpi": 120})
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, top_n * 0.35)))

    colors_instances = plt.cm.Blues([(0.4 + 0.6 * i / top_n) for i in range(top_n)])
    colors_files = plt.cm.Oranges([(0.4 + 0.6 * i / top_n) for i in range(top_n)])

    axes[0].barh(range(top_n), instances, color=colors_instances, edgecolor="white", linewidth=0.5)
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(labels, fontsize=10)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("标注实例数", fontsize=12)
    axes[0].set_title("类别标注实例分布（Top 15）", fontsize=13, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3, linestyle="--")
    for i, v in enumerate(instances):
        axes[0].text(v + 0.3, i, str(v), va="center", fontsize=9, color="#333")

    axes[1].barh(range(top_n), files, color=colors_files, edgecolor="white", linewidth=0.5)
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(labels, fontsize=10)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("文件数", fontsize=12)
    axes[1].set_title("类别文件分布（Top 15）", fontsize=13, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3, linestyle="--")
    for i, v in enumerate(files):
        axes[1].text(v + 0.3, i, str(v), va="center", fontsize=9, color="#333")

    plt.tight_layout(pad=2.0)
    chart_path = str(Path(config.output_dir) / "label_distribution_chart.png")
    Path(chart_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close()
    ui.show_info(f"类别分布图表已保存到: {chart_path}")
else:
    ui.show_warning("无法生成图表 — matplotlib未安装或无统计数据")

if stats_result and stats_result.label_counts:
    summary = stats_result.get_label_summary()
    sorted_summary = sorted(summary.items(), key=lambda x: x[1]["total_instances"], reverse=True)
    print("\\n类别详情（Top 10）:")
    for label, info in sorted_summary[:10]:
        print(f"  {label}: {info['total_instances']}实例 / {info['total_files']}文件 / 范围[{info['min_per_file']}~{info['max_per_file']}]")"""))

# Cell 13: Selection section header (with mode descriptions)
md_cell("sample-section", s("""## 第四部分：样本均衡化选择

基于类别分布的样本挑选机制，支持两种模式：

**n张图片模式 (`n_images`):**
- 类别图片 > n: 按标注数量降序选取前n张不重复图片
- 类别图片 ≤ n: 循环选取直至达到n张（允许重复）

**n个标签样本模式 (`n_labels`):**
- 类别总样本 > n: 按标注数量降序依次选取至累计≥n
- 类别总样本 ≤ n: 循环选取至累计≥n（允许重复）"""))

# Cell 14: Execute selection
code_cell("execute-selection", s("""ui.show_step_header(3, "均衡选择", f"模式: {config.selection_mode}, 目标: {config.target_count}")

selection_result = None
try:
    selection_result = select_balanced_samples(
        source_dir=config.cleaned_dir,
        mode=config.selection_mode,
        target_count=config.target_count,
        random_seed=config.random_seed,
        validate_images=config.validate_images,
        log_file=config.log_file,
        max_workers=config.max_workers,
        use_tqdm=True,
    )

    selection_output = str(Path(config.output_dir) / "selection_result.json")
    Path(selection_output).parent.mkdir(parents=True, exist_ok=True)
    with open(selection_output, "w", encoding="utf-8") as f:
        json.dump(selection_result.to_dict(), f, indent=2, ensure_ascii=False)

    ui.mark_step_completed(3)
    ui.show_result_summary("选择结果", {
        "类别总数": len(selection_result.category_results),
        "总选择图片": selection_result.total_selected_images,
        "唯一图片": selection_result.unique_image_count,
        "结果文件": selection_output,
    })
except Exception as e:
    ui.show_error("均衡选择", str(e))
    selection_result = BalancedSelectionResult(
        source_dir=config.cleaned_dir,
        mode=SelectionMode.N_IMAGES,
        target_count=config.target_count,
    )"""))

# Cell 15: Selection details
md_cell("selection-details", s("""### 各类别选择详情"""))

# Cell 16: Show selection details
code_cell("show-selection-details", s("""if selection_result and selection_result.category_results:
    rows = ""
    for category, sel in sorted(selection_result.category_results.items()):
        dup_str = f"（重复{sel.duplicate_count}）" if sel.has_duplicates else ""
        rows += f"<tr><td style='padding:6px;border:1px solid #c3e6cb;font-weight:bold'>{category}</td>"
        rows += f"<td style='padding:6px;border:1px solid #c3e6cb'>{sel.total_selected_images}{dup_str}</td>"
        rows += f"<td style='padding:6px;border:1px solid #c3e6cb'>{sel.total_selected_labels}</td>"
        rows += f"<td style='padding:6px;border:1px solid #c3e6cb'>{sel.available_images}</td></tr>"

    ui._html(f\"\"\"<div style="{ui._STYLES['config_card']}">
        <h4 style="color:#495057;margin-bottom:8px">📊 各类别选择详情</h4>
        <table style="width:100%;border-collapse:collapse">
            <tr style="background:#e9ecef"><th style="padding:8px;border:1px solid #dee2e6">类别</th><th style="padding:8px;border:1px solid #dee2e6">选择图片</th><th style="padding:8px;border:1px solid #dee2e6">选择标签</th><th style="padding:8px;border:1px solid #dee2e6">可用图片</th></tr>
            {rows}
        </table>
    </div>\"\"\")

    if selection_result.duration:
        ui.show_info(f"选择耗时: {selection_result.duration:.2f}秒")
else:
    ui.show_warning("无选择结果数据")"""))

# Cell 17: Conversion section header
md_cell("convert-section", s("""## 第五部分：数据格式转换

使用 `LabelMeConverter` 将LabelMe标注转换为Unsloth框架兼容格式：

- **标注形状 → 边界框** — 支持polygon/rectangle/circle等形状类型
- **坐标归一化** — 可选归一化至[0,1]范围
- **对话格式生成** — 符合Unsloth多模态训练的messages格式
- **数据集划分** — 自动划分训练集/验证集/测试集并输出JSONL文件"""))

# Cell 18: Execute conversion
code_cell("execute-conversion", s("""ui.show_step_header(4, "格式转换", "转换为Unsloth格式，划分训练/验证/测试集")

conversion_result = None
try:
    conversion_result = convert_to_unsloth_format(
        source_dir=config.cleaned_dir,
        output_dir=config.output_dir,
        instruction_text=config.instruction_text,
        normalize_coordinates=config.normalize_coordinates,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed,
        validate_images=config.validate_images,
        log_file=config.log_file,
        max_workers=config.max_workers,
        use_tqdm=True,
    )

    ui.mark_step_completed(4)
    ui.show_result_summary("转换结果", {
        "总JSON文件": conversion_result.total_json_files,
        "成功转换": f"{conversion_result.converted_count} ({conversion_result.conversion_rate:.1f}%)" if conversion_result.conversion_rate else str(conversion_result.converted_count),
        "转换失败": conversion_result.failed_count,
        "跳过文件": conversion_result.skipped_count,
    })
except Exception as e:
    ui.show_error("格式转换", str(e))
    conversion_result = ConversionResult(
        source_dir=config.cleaned_dir,
        output_dir=config.output_dir,
    )"""))

# Cell 19: Data split & verification
md_cell("split-details", s("""### 数据集划分详情与格式验证

查看训练集/验证集/测试集的统计数据，并验证JSONL格式完整性。"""))

# Cell 20: Show split details
code_cell("show-split-details", s("""if conversion_result:
    split_rows = ""
    for name, split in [("训练集", conversion_result.train_split), ("验证集", conversion_result.val_split), ("测试集", conversion_result.test_split)]:
        if split:
            split_rows += f"<tr><td style='padding:8px;border:1px solid #c3e6cb;font-weight:bold'>{name}</td>"
            split_rows += f"<td style='padding:8px;border:1px solid #c3e6cb'>{split.total_records}</td>"
            split_rows += f"<td style='padding:8px;border:1px solid #c3e6cb'>{split.total_images}</td>"
            split_rows += f"<td style='padding:8px;border:1px solid #c3e6cb'>{split.total_objects}</td>"
            split_rows += f"<td style='padding:8px;border:1px solid #c3e6cb;font-size:11px'>{split.output_path}</td></tr>"

    if split_rows:
        ui._html(f\"\"\"<div style="{ui._STYLES['config_card']}">
            <h4 style="color:#495057;margin-bottom:8px">📊 数据集划分统计</h4>
            <table style="width:100%;border-collapse:collapse">
                <tr style="background:#e9ecef"><th style="padding:8px;border:1px solid #dee2e6">数据集</th><th style="padding:8px;border:1px solid #dee2e6">记录数</th><th style="padding:8px;border:1px solid #dee2e6">图片数</th><th style="padding:8px;border:1px solid #dee2e6">对象数</th><th style="padding:8px;border:1px solid #dee2e6">输出路径</th></tr>
                {split_rows}
            </table>
        </div>\"\"\")

    if conversion_result.duration:
        ui.show_info(f"转换耗时: {conversion_result.duration:.2f}秒")
else:
    ui.show_warning("无转换结果数据")"""))

# Cell 21: Verify format
code_cell("verify-format", s("""def verify_unsloth_format(jsonl_path, num_samples=3):
    \"\"\"验证Unsloth JSONL格式完整性\"\"\"
    path = Path(jsonl_path)
    if not path.exists():
        return False, "文件不存在"

    valid = 0
    total = 0
    errors = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            total += 1
            try:
                record = json.loads(line)
                msgs = record.get("messages", [])
                if "messages" not in record:
                    errors.append(f"样本{i+1}: 缺少messages字段")
                    continue
                if "images" not in record:
                    errors.append(f"样本{i+1}: 缺少images字段")
                    continue
                if len(msgs) != 2 or msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
                    errors.append(f"样本{i+1}: messages格式不符合要求")
                    continue
                valid += 1
            except json.JSONDecodeError as e:
                errors.append(f"样本{i+1}: JSON解析错误 - {e}")

    passed = valid == total
    result_dict = {"检查样本": total, "有效样本": valid, "结果": "✅ 通过" if passed else "❌ 失败"}
    if errors:
        result_dict["错误详情"] = "; ".join(errors[:3])
    return passed, result_dict


ui.show_step_header(5, "输出验证", "验证数据格式完整性")

if conversion_result:
    all_passed = True
    for name, split in [("训练集", conversion_result.train_split), ("验证集", conversion_result.val_split), ("测试集", conversion_result.test_split)]:
        if split and split.output_path:
            passed, info = verify_unsloth_format(split.output_path)
            all_passed = all_passed and passed
            ui.show_result_summary(f"{name}格式验证", info, success=passed)

    ui.mark_step_completed(5 if all_passed else 5)
else:
    ui.show_warning("无转换结果，跳过验证")"""))

# Cell 22: Conversion sample
md_cell("sample-output", s("""### 查看转换示例

展示一条转换后的数据样本格式。"""))

# Cell 23: Show conversion sample
code_cell("show-conversion-sample", s("""if conversion_result and conversion_result.train_split and conversion_result.train_split.output_path:
    jsonl_path = conversion_result.train_split.output_path
    if Path(jsonl_path).exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            line = f.readline()
            if line:
                record = json.loads(line)
                ui._html(f\"\"\"<div style="{ui._STYLES['config_card']}">
                    <h4 style="color:#495057;margin-bottom:12px">📝 转换数据样本示例</h4>
                    <div style="margin-bottom:8px"><b>图像路径:</b> {record.get('images', ['N/A'])[0]}</div>
                    <div style="margin-bottom:8px"><b>用户消息:</b><br><pre style="background:#fff;padding:8px;border-radius:4px;margin:4px 0;font-size:12px;white-space:pre-wrap">{"".join(item.get("text","") for item in record["messages"][0].get("content",[]) if item.get("type")=="text")}</pre></div>
                    <div style="margin-bottom:8px"><b>助手消息:</b><br><pre style="background:#fff;padding:8px;border-radius:4px;margin:4px 0;font-size:12px;white-space:pre-wrap">{"".join(item.get("text","") for item in record["messages"][1].get("content",[]) if item.get("type")=="text")}</pre></div>
                    <div><b>元数据:</b> 尺寸={record.get("metadata",{}).get("image_width",0)}x{record.get("metadata",{}).get("image_height",0)}, 标注数={record.get("metadata",{}).get("num_objects",0)}</div>
                </div>\"\"\")
            else:
                ui.show_warning("训练集文件为空")
    else:
        ui.show_warning(f"文件不存在: {jsonl_path}")
else:
    ui.show_warning("无训练集数据")"""))

# Cell 24: Summary
md_cell("summary-section", s("""## 总结

本Notebook完成了LabelMe标注数据的完整处理流程：

1. **标注清洗** → 验证完整性、过滤无效数据、识别重复标注
2. **统计分析** → 类别分布、标注数量、可视化报告
3. **均衡选择** → 基于类别的均衡采样
4. **格式转换** → Unsloth兼容格式 + 数据集划分
5. **输出验证** → JSONL格式完整性验证

> 📌 后续步骤：使用生成的JSONL文件训练Unsloth模型，根据统计报告调整数据策略"""))

# Cell 25: Final summary
code_cell("final-summary", s("""if cleaning_result and stats_result and selection_result and conversion_result:
    ui.show_final_summary(cleaning_result, stats_result, selection_result, conversion_result, config)
else:
    ui.show_warning("部分步骤执行失败，请检查上方错误信息后重新运行对应单元格")"""))

# Build notebook
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat_minor": 5,
            "pygments_lexer": "ipython3",
            "version": "3.13.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = r"d:\WorkPlace\Pycharm\gemma4-ft\gemma4_multimodal_demo\notebooks\02-data_preparation-labelme_processing.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to {output_path}")
print(f"Total cells: {len(cells)}")