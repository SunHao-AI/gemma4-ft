#!/usr/bin/env python
"""对比两次训练结果与GPU汇总，输出Markdown表格。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_load_gpu_summary(run_dir: Path) -> dict | None:
    summary_files = sorted((run_dir / "gpu_logs").glob("gpu_summary_*.json"))
    if summary_files:
        return load_json(summary_files[-1])
    return None


def resolve_run(run: str) -> tuple[Path, dict, dict | None]:
    path = Path(run)
    result_path = path if path.name == "training_result.json" else path / "training_result.json"
    run_dir = result_path.parent
    result = load_json(result_path)
    gpu_summary = None
    gpu_log_dir = result.get("distributed_config", {}).get("gpu_log_dir")
    if gpu_log_dir:
        summary_files = sorted(Path(gpu_log_dir).glob("gpu_summary_*.json"))
        if summary_files:
            gpu_summary = load_json(summary_files[-1])
    if gpu_summary is None:
        gpu_summary = maybe_load_gpu_summary(run_dir)
    return run_dir, result, gpu_summary


def metric_from_sources(result: dict, gpu_summary: dict | None, key: str, default="N/A"):
    if key in result:
        return result[key]
    if gpu_summary is None:
        return default
    if key == "avg_gpu_util_pct":
        utils = [item.get("avg_utilization_pct", 0) for item in gpu_summary.get("per_gpu_stats", {}).values()]
        return round(sum(utils) / len(utils), 2) if utils else default
    if key == "max_gpu_util_pct":
        utils = [item.get("max_utilization_pct", 0) for item in gpu_summary.get("per_gpu_stats", {}).values()]
        return round(max(utils), 2) if utils else default
    return default


def format_row(label: str, before, after) -> str:
    if isinstance(before, float):
        before = round(before, 4)
    if isinstance(after, float):
        after = round(after, 4)
    delta = "N/A"
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        delta = round(after - before, 4)
    return f"| {label} | {before} | {after} | {delta} |"


def main():
    parser = argparse.ArgumentParser(description="对比两次训练结果")
    parser.add_argument("--before", required=True, help="优化前运行目录或training_result.json路径")
    parser.add_argument("--after", required=True, help="优化后运行目录或training_result.json路径")
    args = parser.parse_args()

    before_dir, before_result, before_gpu = resolve_run(args.before)
    after_dir, after_result, after_gpu = resolve_run(args.after)

    rows = [
        format_row("samples_per_second", metric_from_sources(before_result, before_gpu, "samples_per_second"), metric_from_sources(after_result, after_gpu, "samples_per_second")),
        format_row("steps_per_second", metric_from_sources(before_result, before_gpu, "steps_per_second"), metric_from_sources(after_result, after_gpu, "steps_per_second")),
        format_row("train_runtime_sec", metric_from_sources(before_result, before_gpu, "train_runtime_sec"), metric_from_sources(after_result, after_gpu, "train_runtime_sec")),
        format_row("peak_vram_gb", metric_from_sources(before_result, before_gpu, "peak_vram_gb"), metric_from_sources(after_result, after_gpu, "peak_vram_gb")),
        format_row("vram_utilization_pct", metric_from_sources(before_result, before_gpu, "vram_utilization_pct"), metric_from_sources(after_result, after_gpu, "vram_utilization_pct")),
        format_row("first_step_sec", before_result.get("performance_summary", {}).get("first_step_sec", "N/A"), after_result.get("performance_summary", {}).get("first_step_sec", "N/A")),
        format_row("avg_step_sec", before_result.get("performance_summary", {}).get("avg_step_sec", "N/A"), after_result.get("performance_summary", {}).get("avg_step_sec", "N/A")),
        format_row("dataset_prepare_sec", before_result.get("timings", {}).get("dataset_prepare_sec", "N/A"), after_result.get("timings", {}).get("dataset_prepare_sec", "N/A")),
        format_row("trainer_init_sec", before_result.get("timings", {}).get("trainer_init_sec", "N/A"), after_result.get("timings", {}).get("trainer_init_sec", "N/A")),
        format_row("avg_gpu_util_pct", metric_from_sources(before_result, before_gpu, "avg_gpu_util_pct"), metric_from_sources(after_result, after_gpu, "avg_gpu_util_pct")),
    ]

    report = [
        f"# 训练性能对比",
        "",
        f"- 优化前: `{before_dir}`",
        f"- 优化后: `{after_dir}`",
        "",
        "| 指标 | 优化前 | 优化后 | 差值 |",
        "| --- | ---: | ---: | ---: |",
        *rows,
    ]
    print("\n".join(report))


if __name__ == "__main__":
    main()
