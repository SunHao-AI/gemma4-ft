"""GPU监控工具模块 - 多GPU分布式训练

提供训练过程中各GPU的显存使用、计算负载和温度监控功能。
支持作为HuggingFace Trainer Callback集成，也可独立使用。

特性:
    - 实时GPU显存/利用率/温度监控
    - 分布式环境下仅rank 0记录日志
    - CSV格式日志输出，便于后续分析
    - 与HuggingFace Trainer无缝集成（Callback模式）
"""

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import TrainerCallback

from gemma4_ft.core.runtime import format_file_timestamp, get_aware_now

logger = logging.getLogger(__name__)


class GPUMonitor:
    """GPU监控器 - 采集各GPU的显存、利用率、温度等指标

    采集指标:
        - memory_allocated: 当前显存分配量 (GB)
        - memory_reserved: 当前显存预留量 (GB)
        - memory_total: GPU总显存 (GB)
        - utilization: GPU计算利用率 (%)
        - temperature: GPU温度 (°C)
    """

    def __init__(self, log_dir: str = "gpu_logs", log_interval: int = 50, distributed_rank: Optional[int] = None):
        self.log_dir = Path(log_dir)
        self.log_interval = log_interval
        self._is_main = True
        if distributed_rank is not None:
            self._is_main = distributed_rank == 0
        elif "RANK" in os.environ:
            self._is_main = int(os.environ.get("RANK", 0)) == 0

        self.num_gpus = torch.cuda.device_count()
        self.gpu_names = [torch.cuda.get_device_name(i) for i in range(self.num_gpus)]
        self.gpu_total_memory = [torch.cuda.get_device_properties(i).total_memory / 1024**3 for i in range(self.num_gpus)]

        self._records = []
        self._start_time = None
        self._step_count = 0

        if self._is_main:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = format_file_timestamp()
            self._csv_path = self.log_dir / f"gpu_log_{timestamp}.csv"
            self._summary_path = self.log_dir / f"gpu_summary_{timestamp}.json"
            self._init_csv()
            logger.info(f"GPU监控已启动, 日志目录: {self.log_dir}")

    def _init_csv(self):
        fieldnames = ["timestamp", "step", "elapsed_sec"]
        for i in range(self.num_gpus):
            fieldnames.extend([f"gpu{i}_alloc_gb", f"gpu{i}_reserved_gb", f"gpu{i}_util_pct", f"gpu{i}_temp_c"])
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def _get_nvidia_smi_stats(self):
        stats = {}
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for i, line in enumerate(lines):
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        stats[i] = {"utilization": float(parts[0].strip()), "temperature": float(parts[1].strip())}
        except Exception:
            pass
        return stats

    def snapshot(self, step: int = -1) -> dict:
        """采集当前所有GPU的状态快照"""
        if self._start_time is None:
            self._start_time = time.time()

        elapsed = time.time() - self._start_time
        nvidia_stats = self._get_nvidia_smi_stats() if self._is_main else {}

        record = {
            "timestamp": get_aware_now().isoformat(),
            "step": step if step >= 0 else self._step_count,
            "elapsed_sec": round(elapsed, 2),
        }

        for i in range(self.num_gpus):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3

            nvidia_i = nvidia_stats.get(i, {})
            util = nvidia_i.get("utilization", 0.0)
            temp = nvidia_i.get("temperature", 0.0)

            record[f"gpu{i}_alloc_gb"] = round(alloc, 3)
            record[f"gpu{i}_reserved_gb"] = round(reserved, 3)
            record[f"gpu{i}_util_pct"] = round(util, 1)
            record[f"gpu{i}_temp_c"] = round(temp, 1)

        self._records.append(record)
        self._step_count += 1
        return record

    def log_snapshot(self, step: int = -1) -> None:
        """采集并记录快照到CSV"""
        if not self._is_main:
            return

        record = self.snapshot(step)

        fieldnames = ["timestamp", "step", "elapsed_sec"]
        for i in range(self.num_gpus):
            fieldnames.extend([f"gpu{i}_alloc_gb", f"gpu{i}_reserved_gb", f"gpu{i}_util_pct", f"gpu{i}_temp_c"])

        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(record)

    def print_summary(self, step: int = -1) -> None:
        """打印当前GPU状态摘要"""
        if not self._is_main:
            return

        record = self.snapshot(step)
        step_num = record["step"]
        elapsed = record["elapsed_sec"]

        print(f"\n[GPU监控] Step {step_num}, 耗时 {elapsed:.1f}s")
        print(f"{'GPU':>4} | {'分配(GB)':>9} | {'预留(GB)':>9} | {'利用率%':>8} | {'温度°C':>7} | {'总显存(GB)':>10}")
        print("-" * 65)

        total_alloc = 0
        total_util = 0
        for i in range(self.num_gpus):
            alloc = record[f"gpu{i}_alloc_gb"]
            reserved = record[f"gpu{i}_reserved_gb"]
            util = record[f"gpu{i}_util_pct"]
            temp = record[f"gpu{i}_temp_c"]
            total_mem = self.gpu_total_memory[i]
            total_alloc += alloc
            total_util += util
            print(f"  {i:>2} | {alloc:>9.3f} | {reserved:>9.3f} | {util:>8.1f} | {temp:>7.1f} | {total_mem:>10.1f}")

        avg_alloc = total_alloc / self.num_gpus
        avg_util = total_util / self.num_gpus
        print(f"  平均 | {avg_alloc:>9.3f} | {'':>9} | {avg_util:>8.1f} | {'':>7} | {'':>10}")

    def save_summary(self) -> dict:
        """保存训练结束后的汇总统计"""
        if not self._is_main or not self._records:
            return {}

        summary = {
            "gpu_count": self.num_gpus,
            "gpu_names": self.gpu_names,
            "gpu_total_memory_gb": self.gpu_total_memory,
            "total_steps_logged": len(self._records),
            "total_elapsed_sec": self._records[-1]["elapsed_sec"] if self._records else 0,
            "per_gpu_stats": {},
        }

        for i in range(self.num_gpus):
            allocs = [r[f"gpu{i}_alloc_gb"] for r in self._records]
            reserved = [r[f"gpu{i}_reserved_gb"] for r in self._records]
            utils = [r[f"gpu{i}_util_pct"] for r in self._records]
            temps = [r[f"gpu{i}_temp_c"] for r in self._records]

            summary["per_gpu_stats"][f"gpu{i}"] = {
                "name": self.gpu_names[i],
                "total_memory_gb": self.gpu_total_memory[i],
                "avg_alloc_gb": round(sum(allocs) / len(allocs), 3),
                "max_alloc_gb": round(max(allocs), 3),
                "min_alloc_gb": round(min(allocs), 3),
                "avg_reserved_gb": round(sum(reserved) / len(reserved), 3),
                "max_reserved_gb": round(max(reserved), 3),
                "avg_utilization_pct": round(sum(utils) / len(utils), 1),
                "max_utilization_pct": round(max(utils), 1),
                "avg_temperature_c": round(sum(temps) / len(temps), 1),
                "max_temperature_c": round(max(temps), 1),
                "memory_efficiency_pct": round(max(allocs) / self.gpu_total_memory[i] * 100, 1),
            }

        with open(self._summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"GPU监控汇总已保存到: {self._summary_path}")
        logger.info(f"详细日志: {self._csv_path}")
        return summary


class GPUMonitorCallback(TrainerCallback):
    """HuggingFace Trainer Callback - 自动集成GPU监控

    仅记录CSV日志和JSON汇总, 不输出GPU监控表格。
    用户可自行使用nvitop等工具查看GPU实时状态。

    使用方式:
        monitor = GPUMonitor(log_dir="gpu_logs", log_interval=50)
        callback = GPUMonitorCallback(monitor)
        trainer = SFTTrainer(..., callbacks=[callback])
    """

    def __init__(self, gpu_monitor: GPUMonitor, print_interval: int = 100):
        self.monitor = gpu_monitor
        self.print_interval = print_interval

    def on_train_begin(self, args, state, control, **kwargs):
        self.monitor.log_snapshot(step=0)

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % self.monitor.log_interval == 0:
            self.monitor.log_snapshot(step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self.monitor.log_snapshot(step=state.global_step)
        self.monitor.save_summary()


def benchmark_single_vs_multi(
    model_fn,
    data_fn,
    train_config_single: dict,
    train_config_multi: dict,
    num_gpus: int = 8,
    output_dir: str = "benchmark_results",
):
    """单GPU vs 多GPU性能对比基准测试

    Args:
        model_fn: 模型加载函数 (返回model, tokenizer/processor)
        data_fn: 数据加载函数 (返回dataset)
        train_config_single: 单GPU训练配置
        train_config_multi: 多GPU训练配置
        num_gpus: 多GPU数量
        output_dir: 结果输出目录

    Returns:
        dict: 包含对比数据的字典
    """
    from trl import SFTTrainer, SFTConfig

    results = {"single_gpu": {}, "multi_gpu": {}, "comparison": {}}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # === 单GPU基准 ===
    print("=" * 60)
    print("1. 单GPU基准测试")
    print("=" * 60)

    model, tokenizer = model_fn()

    monitor_single = GPUMonitor(
        log_dir=str(output_path / "single_gpu_logs"),
        log_interval=10,
        distributed_rank=0,
    )

    from unsloth.trainer import UnslothVisionDataCollator

    trainer_single = SFTTrainer(
        model=model,
        train_dataset=data_fn(),
        processing_class=tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer) if hasattr(tokenizer, "image_processor") else None,
        args=SFTConfig(**train_config_single, max_seq_length=train_config_single.get("max_seq_length", 2048)),
        callbacks=[GPUMonitorCallback(monitor_single)],
    )

    start_time = time.time()
    single_stats = trainer_single.train()
    single_time = time.time() - start_time

    single_metrics = single_stats.metrics
    results["single_gpu"] = {
        "train_time_sec": round(single_time, 2),
        "train_loss": single_metrics.get("train_loss", 0),
        "train_runtime": single_metrics.get("train_runtime", 0),
        "train_samples_per_second": single_metrics.get("train_samples_per_second", 0),
        "train_steps_per_second": single_metrics.get("train_steps_per_second", 0),
    }

    single_summary = monitor_single.save_summary()
    if single_summary:
        results["single_gpu"]["gpu_summary"] = single_summary

    del model, trainer_single
    torch.cuda.empty_cache()

    # === 多GPU基准 (需要在torchrun环境下运行) ===
    print("\n" + "=" * 60)
    print("2. 多GPU基准测试")
    print("=" * 60)
    print(f"注意: 多GPU基准测试需要在torchrun/accelerate环境下执行")
    print(f"请使用以下命令运行:")
    print(f"  torchrun --nproc_per_node={num_gpus} train_distributed.py --benchmark ...")

    results["multi_gpu"] = {"note": "需在torchrun环境下执行多GPU基准"}

    # === 对比计算 ===
    if "train_time_sec" in results["single_gpu"] and "train_time_sec" in results["multi_gpu"]:
        speedup = results["single_gpu"]["train_time_sec"] / results["multi_gpu"]["train_time_sec"]
        results["comparison"] = {
            "speedup_ratio": round(speedup, 2),
            "time_saved_pct": round((1 - 1 / speedup) * 100, 1),
            "single_gpu_time_sec": results["single_gpu"]["train_time_sec"],
            "multi_gpu_time_sec": results["multi_gpu"]["train_time_sec"],
        }

    with open(output_path / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n基准测试结果已保存到: {output_path / 'benchmark_results.json'}")
    return results


def print_gpu_info():
    """打印当前环境所有GPU的详细信息"""
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        print("未检测到GPU！")
        return

    print(f"检测到 {num_gpus} 张GPU:")
    print(f"{'GPU':>4} | {'名称':>30} | {'总显存(GB)':>10} | {'CUDA计算能力':>14} | {'BF16支持':>8}")
    print("-" * 80)

    bf16_supported = torch.cuda.is_bf16_supported()

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        name = props.name
        total_mem = props.total_memory / 1024**3
        major = props.major
        minor = props.minor
        print(f"  {i:>2} | {name:>30} | {total_mem:>10.1f} | {major}.{minor:>13} | {'✓' if bf16_supported else '✗':>8}")

    print(f"\nBF16混合精度: {'可用' if bf16_supported else '不可用 (将使用FP16)'}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"NCCL可用: {'是' if torch.distributed.is_nccl_available() else '否'}")
