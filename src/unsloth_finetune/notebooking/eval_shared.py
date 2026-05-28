from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import requests
from PIL import Image

from unsloth_finetune.data.labelme.detection_format import parse_box_2d_json_ground_truth


class IOUCalculator:
    @staticmethod
    def calculate_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def calculate_batch_iou(detections1: List[Dict], detections2: List[Dict]) -> Dict[str, Any]:
        if not detections1 or not detections2:
            return {
                "mean_iou": 0.0,
                "max_iou": 0.0,
                "min_iou": 0.0,
                "num_pairs": 0,
                "class_ious": {},
            }

        all_ious = []
        class_ious: Dict[str, List[float]] = {}
        for detection in detections1:
            bbox1 = detection.get("bbox", [0, 0, 0, 0])
            label = detection.get("label", "unknown")
            best_iou = 0.0
            for other in detections2:
                best_iou = max(best_iou, IOUCalculator.calculate_iou(bbox1, other.get("bbox", [0, 0, 0, 0])))
            if best_iou > 0:
                all_ious.append(best_iou)
                class_ious.setdefault(label, []).append(best_iou)

        class_stats = {}
        for label, ious in class_ious.items():
            class_stats[label] = {
                "mean_iou": float(np.mean(ious)),
                "max_iou": float(np.max(ious)),
                "min_iou": float(np.min(ious)),
                "count": len(ious),
            }

        return {
            "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
            "max_iou": float(np.max(all_ious)) if all_ious else 0.0,
            "min_iou": float(np.min(all_ious)) if all_ious else 0.0,
            "num_pairs": len(all_ious),
            "class_ious": class_stats,
        }

    @staticmethod
    def filter_by_confidence(detections: List[Dict], threshold: float) -> List[Dict]:
        return [det for det in detections if det.get("confidence", 0) >= threshold]


class MetricsCalculator:
    DEFAULT_IOU_THRESHOLD = 0.5

    @staticmethod
    def compute_sample_metrics(
        detections: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        if not ground_truth:
            if not detections:
                return {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "num_det": 0,
                    "num_gt": 0,
                    "num_match": 0,
                    "mean_match_iou": 0.0,
                    "det_success": True,
                }
            return {
                "precision": 0.0,
                "recall": 1.0,
                "f1": 0.0,
                "num_det": len(detections),
                "num_gt": 0,
                "num_match": 0,
                "mean_match_iou": 0.0,
                "det_success": False,
            }

        if not detections:
            return {
                "precision": 1.0,
                "recall": 0.0,
                "f1": 0.0,
                "num_det": 0,
                "num_gt": len(ground_truth),
                "num_match": 0,
                "mean_match_iou": 0.0,
                "det_success": False,
            }

        matched_gt = set()
        matched_det = set()
        match_ious = []
        for index, detection in enumerate(detections):
            det_bbox = detection.get("bbox", [0, 0, 0, 0])
            best_iou = 0.0
            best_gt_index = -1
            for gt_index, ground_truth_item in enumerate(ground_truth):
                if gt_index in matched_gt:
                    continue
                gt_bbox = ground_truth_item.get("bbox", [0, 0, 0, 0])
                iou = IOUCalculator.calculate_iou(det_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index
            if best_iou >= iou_threshold and best_gt_index >= 0:
                matched_gt.add(best_gt_index)
                matched_det.add(index)
                match_ious.append(best_iou)

        precision = len(matched_det) / len(detections)
        recall = len(matched_gt) / len(ground_truth)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        det_success = len(matched_gt) == len(ground_truth) and len(matched_det) == len(detections)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_det": len(detections),
            "num_gt": len(ground_truth),
            "num_match": len(matched_det),
            "mean_match_iou": float(np.mean(match_ious)) if match_ious else 0.0,
            "det_success": det_success,
        }

    @staticmethod
    def aggregate_metrics(sample_metrics_list: List[Dict]) -> Dict[str, Any]:
        if not sample_metrics_list:
            return {}

        keys = ["precision", "recall", "f1", "num_det", "num_gt", "num_match", "mean_match_iou"]
        result = {}
        for key in keys:
            values = [metric[key] for metric in sample_metrics_list if key in metric]
            if values:
                result[f"mean_{key}"] = float(np.mean(values))
                result[f"std_{key}"] = float(np.std(values))
            else:
                result[f"mean_{key}"] = 0.0
                result[f"std_{key}"] = 0.0

        result["total_samples"] = len(sample_metrics_list)
        result["success_rate"] = float(np.mean([metric.get("det_success", False) for metric in sample_metrics_list]))
        return result


class DatasetLoader:
    @staticmethod
    def load_jsonl(filepath: str) -> List[Dict]:
        records = []
        with open(filepath, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    @staticmethod
    def parse_ground_truth(record: Dict) -> List[Dict]:
        import re

        metadata = record.get("metadata", {})
        img_width = metadata.get("image_width", 1000)
        img_height = metadata.get("image_height", 1000)

        assistant_text = ""
        for message in record.get("messages", []):
            if message.get("role") != "assistant":
                continue
            for item in message.get("content", []):
                if item.get("type") == "text":
                    assistant_text += item.get("text", "")

        if not assistant_text:
            return []

        output_format = metadata.get("output_format", "labelme_text")

        # Try box_2d_json format first
        if output_format == "box_2d_json" or assistant_text.strip().startswith("["):
            json_detections = parse_box_2d_json_ground_truth(assistant_text, img_width, img_height)
            if json_detections:
                return json_detections

        # Fall back to legacy regex
        pattern = r"-\s*(\S+)\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
        gt_bboxes = []
        for match in re.finditer(pattern, assistant_text):
            label = match.group(1)
            coords = [float(match.group(index)) for index in range(2, 6)]
            x_min, y_min, x_max, y_max = coords
            if all(0 <= coord <= 1 for coord in coords):
                x_min, y_min, x_max, y_max = (
                    int(x_min * img_width),
                    int(y_min * img_height),
                    int(x_max * img_width),
                    int(y_max * img_height),
                )
            else:
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            gt_bboxes.append(
                {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "label": label,
                    "confidence": 1.0,
                }
            )
        return gt_bboxes

    @staticmethod
    def extract_query(record: Dict) -> str:
        for message in record.get("messages", []):
            if message.get("role") != "user":
                continue
            for item in message.get("content", []):
                if item.get("type") == "text":
                    return item.get("text", "")
        return ""

    @staticmethod
    def extract_image_path(record: Dict) -> str:
        images = record.get("images", [])
        if images:
            return images[0]
        return record.get("metadata", {}).get("json_path", "")

    @staticmethod
    def load_image(image_path: str) -> Optional[Image.Image]:
        try:
            if image_path.startswith(("http://", "https://")):
                response = requests.get(image_path, timeout=30)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")

            path = Path(image_path)
            if path.exists():
                return Image.open(path).convert("RGB")
            return None
        except Exception:
            return None


class ResultManager:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def _model_cache_dir(self, model_key: str) -> Path:
        model_dir = self.result_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def save_sample_results(self, model_key: str, split_name: str, results: list) -> None:
        path = self._model_cache_dir(model_key) / f"{split_name}_samples.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"已保存 {model_key}/{split_name} 样本结果: {path}")

    def save_aggregated_metrics(self, model_key: str, split_name: str, metrics: dict) -> None:
        path = self._model_cache_dir(model_key) / f"{split_name}_aggregated.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        print(f"已保存 {model_key}/{split_name} 汇总指标: {path}")

    def load_sample_results(self, model_key: str, split_name: str) -> Optional[list]:
        path = self._model_cache_dir(model_key) / f"{split_name}_samples.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_aggregated_metrics(self, model_key: str, split_name: str) -> Optional[dict]:
        path = self._model_cache_dir(model_key) / f"{split_name}_aggregated.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def has_cached_results(self, model_key: str, split_name: str) -> bool:
        samples_path = self._model_cache_dir(model_key) / f"{split_name}_samples.json"
        agg_path = self._model_cache_dir(model_key) / f"{split_name}_aggregated.json"
        return samples_path.exists() and agg_path.exists()

    def clear_cache(self, model_key: Optional[str] = None, split_name: Optional[str] = None) -> None:
        if model_key:
            if split_name:
                for suffix in ["_samples.json", "_aggregated.json"]:
                    path = self._model_cache_dir(model_key) / f"{split_name}{suffix}"
                    if path.exists():
                        path.unlink()
                return

            for path in self._model_cache_dir(model_key).glob("*.json"):
                path.unlink()
            return

        for path in self.result_dir.glob("**/*.json"):
            path.unlink()


def aggregate_metric_dicts(sample_metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not sample_metrics_list:
        return {}

    keys = ["precision", "recall", "f1", "num_det", "num_gt", "num_match", "mean_match_iou"]
    result = {}
    for key in keys:
        values = [float(metric[key]) for metric in sample_metrics_list if key in metric]
        if not values:
            result[f"mean_{key}"] = 0.0
            result[f"std_{key}"] = 0.0
            continue
        mean_value = sum(values) / len(values)
        result[f"mean_{key}"] = mean_value
        result[f"std_{key}"] = pstdev(values) if len(values) > 1 else 0.0

    success_flags = [1.0 if metric.get("det_success", False) else 0.0 for metric in sample_metrics_list]
    result["total_samples"] = len(sample_metrics_list)
    result["success_rate"] = sum(success_flags) / len(success_flags) if success_flags else 0.0
    return result


def _display_width(value: str) -> int:
    width = 0
    for character in value:
        width += 2 if "\u4e00" <= character <= "\u9fff" else 1
    return width


def _pad_to_width(value: str, target_width: int, align: str = "left") -> str:
    current = _display_width(value)
    padding = target_width - current
    if padding <= 0:
        return value[: target_width // 2 if any("\u4e00" <= char <= "\u9fff" for char in value) else target_width]
    if align == "right":
        return " " * padding + value
    if align == "center":
        left_pad = padding // 2
        return " " * left_pad + value + " " * (padding - left_pad)
    return value + " " * padding


def format_metrics_table(all_metrics: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    col_widths = {
        "split": 8,
        "model": 8,
        "precision": 10,
        "recall": 10,
        "f1": 8,
        "iou": 10,
        "success": 10,
        "samples": 8,
    }
    header_parts = [
        _pad_to_width("数据集", col_widths["split"]),
        _pad_to_width("模型", col_widths["model"]),
        _pad_to_width("精确率", col_widths["precision"]),
        _pad_to_width("召回率", col_widths["recall"]),
        _pad_to_width("F1", col_widths["f1"]),
        _pad_to_width("匹配IOU", col_widths["iou"]),
        _pad_to_width("成功率", col_widths["success"]),
        _pad_to_width("样本数", col_widths["samples"]),
    ]
    header = " | ".join(header_parts)
    sep = "-" * len(header)
    double_sep = "=" * len(header)
    lines = [double_sep, header, sep]

    for split_name in ["train", "valid", "test"]:
        if split_name not in all_metrics:
            continue
        split_data = all_metrics[split_name]
        for model_key, model_label in [("base", "原始"), ("finetuned", "微调")]:
            metrics = split_data.get(model_key, {})
            line = (
                f"{split_name:<8} | {model_label:<8} | "
                f"{metrics.get('mean_precision', 0):.3f}    | "
                f"{metrics.get('mean_recall', 0):.3f}    | "
                f"{metrics.get('mean_f1', 0):.3f}    | "
                f"{metrics.get('mean_mean_match_iou', 0):.3f}    | "
                f"{metrics.get('success_rate', 0):.3f}    | "
                f"{metrics.get('total_samples', 0):<6}"
            )
            lines.append(line)
        lines.append(sep)

    return "\n".join(lines)


def compute_diff_table(all_metrics: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    header = f"{'数据集':<8} | {'精确率差':<10} | {'召回率差':<10} | {'F1差':<10} | {'IOU差':<10} | {'成功率差':<10}"
    sep = "-" * len(header)
    lines = ["\n微调模型改进幅度 (微调 - 原始):", sep, header, sep]

    for split_name in ["train", "valid", "test"]:
        if split_name not in all_metrics:
            continue
        base = all_metrics[split_name].get("base", {})
        finetuned = all_metrics[split_name].get("finetuned", {})
        diffs = {
            "precision": finetuned.get("mean_precision", 0) - base.get("mean_precision", 0),
            "recall": finetuned.get("mean_recall", 0) - base.get("mean_recall", 0),
            "f1": finetuned.get("mean_f1", 0) - base.get("mean_f1", 0),
            "iou": finetuned.get("mean_mean_match_iou", 0) - base.get("mean_mean_match_iou", 0),
            "success": finetuned.get("success_rate", 0) - base.get("success_rate", 0),
        }
        line = f"{split_name:<8} | {diffs['precision']:>+8.3f}  | {diffs['recall']:>+8.3f}  | " f"{diffs['f1']:>+8.3f}  | {diffs['iou']:>+8.3f}  | {diffs['success']:>+8.3f}"
        lines.append(line)

    lines.append(sep)
    return "\n".join(lines)


def generate_analysis(
    all_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    iou_match_threshold: float,
) -> str:
    if not all_metrics:
        return "暂无评估结果"

    report_lines = [
        "=" * 60,
        "微调效果差异分析报告",
        "=" * 60,
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"IOU匹配阈值: {iou_match_threshold}",
        "",
        "一、总体改进分析",
        "-" * 40,
    ]

    all_improvements = []
    all_degradations = []
    metric_names = {
        "mean_precision": "精确率",
        "mean_recall": "召回率",
        "mean_f1": "F1分数",
        "mean_mean_match_iou": "匹配IOU",
        "success_rate": "检测成功率",
    }

    for split_name in ["train", "valid", "test"]:
        if split_name not in all_metrics:
            continue
        base = all_metrics[split_name].get("base", {})
        finetuned = all_metrics[split_name].get("finetuned", {})
        report_lines.append(f"\n  [{split_name} 数据集]")
        for key, label in metric_names.items():
            base_val = base.get(key, 0)
            finetuned_val = finetuned.get(key, 0)
            diff = finetuned_val - base_val
            if diff > 0.01:
                report_lines.append(f"    + {label}: {base_val:.3f} -> {finetuned_val:.3f} (提升 {diff:+.3f})")
                all_improvements.append((split_name, label, diff))
            elif diff < -0.01:
                report_lines.append(f"    - {label}: {base_val:.3f} -> {finetuned_val:.3f} (退化 {diff:+.3f})")
                all_degradations.append((split_name, label, diff))
            else:
                report_lines.append(f"    = {label}: {base_val:.3f} -> {finetuned_val:.3f} (持平)")

    report_lines.extend(["", "", "二、关键发现", "-" * 40])
    if all_improvements:
        report_lines.append("\n  最显著改进:")
        for split_name, label, diff in sorted(all_improvements, key=lambda item: item[2], reverse=True)[:3]:
            report_lines.append(f"    - {split_name}数据集的{label}: 提升{diff:+.3f}")
    if all_degradations:
        report_lines.append("\n  需关注退化:")
        for split_name, label, diff in sorted(all_degradations, key=lambda item: item[2])[:3]:
            report_lines.append(f"    - {split_name}数据集的{label}: 退化{diff:+.3f}")

    report_lines.extend(["", "", "三、泛化能力分析", "-" * 40])
    splits_available = [split_name for split_name in ["train", "valid", "test"] if split_name in all_metrics]
    if len(splits_available) >= 2:
        train_ft = all_metrics.get("train", {}).get("finetuned", {}).get("mean_f1", 0)
        valid_ft = all_metrics.get("valid", {}).get("finetuned", {}).get("mean_f1", 0)
        test_ft = all_metrics.get("test", {}).get("finetuned", {}).get("mean_f1", 0)
        train_base = all_metrics.get("train", {}).get("base", {}).get("mean_f1", 0)
        test_base = all_metrics.get("test", {}).get("base", {}).get("mean_f1", 0)
        ft_gap = train_ft - test_ft if train_ft and test_ft else 0
        base_gap = train_base - test_base if train_base and test_base else 0

        report_lines.append(f"\n  微调模型 F1: train={train_ft:.3f}, valid={valid_ft:.3f}, test={test_ft:.3f}")
        report_lines.append(f"  原始模型 F1: train={train_base:.3f}, test={test_base:.3f}")
        report_lines.append(f"\n  微调模型 train-test差距: {ft_gap:+.3f}")
        report_lines.append(f"  原始模型 train-test差距: {base_gap:+.3f}")

        if ft_gap < base_gap:
            report_lines.append("\n  结论: 微调模型的泛化差距更小, 泛化能力更优")
        elif ft_gap > base_gap:
            report_lines.append("\n  结论: 微调模型的泛化差距更大, 可能存在过拟合风险")
        else:
            report_lines.append("\n  结论: 两个模型的泛化差距相近")

    report_lines.extend(
        [
            "",
            "",
            "四、后续优化建议",
            "-" * 40,
            "  - 增加IOU阈值(0.5->0.7)可更严格评估定位精度",
            "  - 可按类别分别统计指标, 找出薄弱类别针对性优化",
        ]
    )
    if all_degradations:
        report_lines.insert(-2, "  - 关注退化指标, 考虑调整训练数据分布或LoRA参数")

    return "\n".join(report_lines)


def infer_preferred_split_name(split_names: Iterable[str]) -> str:
    names = list(split_names)
    for preferred in ["test", "valid", "train"]:
        if preferred in names:
            return preferred
    return names[0] if names else "test"


def load_multi_gpu_evaluation_outputs(
    datasets: Dict[str, List[Dict]],
    result_dir: str,
    split_names: Optional[Iterable[str]] = None,
    max_samples: Optional[int] = None,
) -> tuple[Dict[str, List[Dict]], Dict[str, Dict[str, Dict[str, Any]]]]:
    result_dir_path = Path(result_dir)
    ft_path = result_dir_path / "finetuned_results.json"
    base_path = result_dir_path / "base_results.json"
    summary_path = result_dir_path / "comparison_summary.json"

    missing_files = [str(path) for path in [ft_path, base_path] if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"多GPU结果文件不存在: {missing_files}")

    split_name = infer_preferred_split_name(split_names or datasets.keys())
    records = datasets.get(split_name, [])
    eval_records = records[:max_samples] if max_samples and max_samples < len(records) else records

    with open(ft_path, "r", encoding="utf-8") as handle:
        ft_results = json.load(handle)
    with open(base_path, "r", encoding="utf-8") as handle:
        base_results = json.load(handle)

    summary = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)

    def build_buckets(results: List[Dict]) -> Dict[tuple[str, str], List[Dict]]:
        buckets: Dict[tuple[str, str], List[Dict]] = {}
        for item in results:
            key = (item.get("image_path", ""), item.get("query", ""))
            buckets.setdefault(key, []).append(item)
        return buckets

    ft_buckets = build_buckets(ft_results)
    base_buckets = build_buckets(base_results)

    combined_results = []
    missing_pairs = 0
    for index, record in enumerate(eval_records):
        image_path = DatasetLoader.extract_image_path(record)
        query = DatasetLoader.extract_query(record)
        key = (image_path, query)

        ft_candidates = ft_buckets.get(key, [])
        base_candidates = base_buckets.get(key, [])
        if not ft_candidates or not base_candidates:
            missing_pairs += 1
            continue

        ft_item = ft_candidates.pop(0)
        base_item = base_candidates.pop(0)
        det_base = base_item.get("detections", [])
        det_ft = ft_item.get("detections", [])

        combined_results.append(
            {
                "index": index,
                "split": split_name,
                "image_path": image_path,
                "query": query,
                "ground_truth": DatasetLoader.parse_ground_truth(record),
                "det_base": det_base,
                "det_ft": det_ft,
                "base_metrics": base_item.get("metrics", {}),
                "ft_metrics": ft_item.get("metrics", {}),
                "model_iou": IOUCalculator.calculate_batch_iou(det_base, det_ft),
            }
        )

    leftover_ft = sum(len(items) for items in ft_buckets.values())
    leftover_base = sum(len(items) for items in base_buckets.values())
    if missing_pairs or leftover_ft or leftover_base:
        print("警告: 多GPU结果与数据集对齐存在差异, " f"missing_pairs={missing_pairs}, leftover_ft={leftover_ft}, leftover_base={leftover_base}")

    base_metric_list = [item["base_metrics"] for item in combined_results]
    ft_metric_list = [item["ft_metrics"] for item in combined_results]
    base_agg = summary.get("base") or MetricsCalculator.aggregate_metrics(base_metric_list)
    ft_agg = summary.get("finetuned") or MetricsCalculator.aggregate_metrics(ft_metric_list)

    return {split_name: combined_results}, {split_name: {"base": base_agg, "finetuned": ft_agg}}


class MetricsVisualizer:
    MODEL_COLORS = {"原始模型": "#4C72B0", "微调模型": "#DD8452"}
    SPLIT_COLORS = {"train": "#4C72B0", "valid": "#55A868", "test": "#C44E52"}

    @staticmethod
    def _ensure_chinese_font(plt_module) -> dict:
        font_family = plt_module.rcParams.get("font.sans-serif", ["sans-serif"])
        if isinstance(font_family, list):
            font_name = font_family[0] if font_family else "sans-serif"
        else:
            font_name = str(font_family)
        font_name = font_name if font_name and font_name != "Arial" else "sans-serif"
        return {"fontname": font_name, "fontsize": 12}

    @staticmethod
    def plot_metrics_bar_chart(plt_module, metrics_dict: Dict, title: str = "模型性能对比") -> None:
        font_props = MetricsVisualizer._ensure_chinese_font(plt_module)
        metric_keys = ["mean_precision", "mean_recall", "mean_f1", "mean_mean_match_iou", "success_rate"]
        metric_labels = ["精确率", "召回率", "F1分数", "平均匹配IOU", "检测成功率"]
        splits = list(metrics_dict.keys())
        n_metrics = len(metric_keys)
        n_splits = len(splits)

        figure, axes = plt_module.subplots(1, n_splits, figsize=(5 * n_splits, 6), sharey=True)
        if n_splits == 1:
            axes = [axes]

        for axis_index, split_name in enumerate(splits):
            axis = axes[axis_index]
            split_data = metrics_dict[split_name]
            x_values = np.arange(n_metrics)
            width = 0.35

            base_vals = [split_data.get("base", {}).get(key, 0.0) for key in metric_keys]
            ft_vals = [split_data.get("finetuned", {}).get(key, 0.0) for key in metric_keys]

            bars1 = axis.bar(
                x_values - width / 2,
                base_vals,
                width,
                label="原始模型",
                color=MetricsVisualizer.MODEL_COLORS["原始模型"],
                alpha=0.85,
            )
            bars2 = axis.bar(
                x_values + width / 2,
                ft_vals,
                width,
                label="微调模型",
                color=MetricsVisualizer.MODEL_COLORS["微调模型"],
                alpha=0.85,
            )

            axis.set_title(f"{split_name} 数据集", fontsize=13, fontweight="bold", fontname=font_props["fontname"])
            axis.set_xticks(x_values)
            axis.set_xticklabels(metric_labels, fontsize=9, fontname=font_props["fontname"])
            axis.set_ylim(0, 1.05)
            axis.legend(fontsize=9, prop={"family": font_props["fontname"]})
            axis.grid(axis="y", alpha=0.3)

            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                if height > 0.01:
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.02,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontname=font_props["fontname"],
                    )

        figure.suptitle(title, fontsize=15, fontweight="bold", y=1.02, fontname=font_props["fontname"])
        plt_module.tight_layout()
        plt_module.show()

    @staticmethod
    def plot_radar_chart(plt_module, metrics_dict: Dict, title: str = "模型性能雷达图") -> None:
        font_props = MetricsVisualizer._ensure_chinese_font(plt_module)
        metric_keys = ["mean_precision", "mean_recall", "mean_f1", "mean_mean_match_iou", "success_rate"]
        metric_labels = ["精确率", "召回率", "F1", "匹配IOU", "成功率"]
        splits = list(metrics_dict.keys())
        n_splits = len(splits)

        figure, axes = plt_module.subplots(1, n_splits, figsize=(6 * n_splits, 6), subplot_kw=dict(polar=True))
        if n_splits == 1:
            axes = [axes]

        angles = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False).tolist()
        angles += angles[:1]

        for axis_index, split_name in enumerate(splits):
            axis = axes[axis_index]
            split_data = metrics_dict[split_name]

            base_vals = [split_data.get("base", {}).get(key, 0.0) for key in metric_keys]
            ft_vals = [split_data.get("finetuned", {}).get(key, 0.0) for key in metric_keys]
            base_vals += base_vals[:1]
            ft_vals += ft_vals[:1]

            axis.plot(
                angles,
                base_vals,
                "o-",
                linewidth=2,
                label="原始模型",
                color=MetricsVisualizer.MODEL_COLORS["原始模型"],
            )
            axis.fill(angles, base_vals, alpha=0.15, color=MetricsVisualizer.MODEL_COLORS["原始模型"])
            axis.plot(
                angles,
                ft_vals,
                "o-",
                linewidth=2,
                label="微调模型",
                color=MetricsVisualizer.MODEL_COLORS["微调模型"],
            )
            axis.fill(angles, ft_vals, alpha=0.15, color=MetricsVisualizer.MODEL_COLORS["微调模型"])

            axis.set_xticks(angles[:-1])
            axis.set_xticklabels(metric_labels, fontsize=10, fontname=font_props["fontname"])
            axis.set_ylim(0, 1.0)
            axis.set_title(f"{split_name} 数据集", fontsize=12, fontweight="bold", pad=20, fontname=font_props["fontname"])
            axis.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9, prop={"family": font_props["fontname"]})

        figure.suptitle(title, fontsize=15, fontweight="bold", y=1.05, fontname=font_props["fontname"])
        plt_module.tight_layout()
        plt_module.show()

    @staticmethod
    def plot_diff_heatmap(plt_module, metrics_dict: Dict, title: str = "微调改进幅度热力图") -> None:
        font_props = MetricsVisualizer._ensure_chinese_font(plt_module)
        metric_keys = ["mean_precision", "mean_recall", "mean_f1", "mean_mean_match_iou", "success_rate"]
        metric_labels = ["精确率", "召回率", "F1", "匹配IOU", "成功率"]
        splits = list(metrics_dict.keys())

        diff_matrix = np.zeros((len(splits), len(metric_keys)))
        for row_index, split_name in enumerate(splits):
            split_data = metrics_dict[split_name]
            for column_index, key in enumerate(metric_keys):
                base_val = split_data.get("base", {}).get(key, 0.0)
                ft_val = split_data.get("finetuned", {}).get(key, 0.0)
                diff_matrix[row_index, column_index] = ft_val - base_val

        figure, axis = plt_module.subplots(figsize=(8, 4))
        image = axis.imshow(diff_matrix, cmap="RdYlGn", aspect="auto", vmin=-0.3, vmax=0.3)

        axis.set_xticks(np.arange(len(metric_keys)))
        axis.set_xticklabels(metric_labels, fontsize=10, fontname=font_props["fontname"])
        axis.set_yticks(np.arange(len(splits)))
        axis.set_yticklabels(splits, fontsize=10, fontname=font_props["fontname"])

        for row_index in range(len(splits)):
            for column_index in range(len(metric_keys)):
                value = diff_matrix[row_index, column_index]
                color = "white" if abs(value) > 0.15 else "black"
                axis.text(column_index, row_index, f"{value:+.3f}", ha="center", va="center", fontsize=9, color=color, fontname=font_props["fontname"])

        axis.set_title(title, fontsize=13, fontweight="bold", fontname=font_props["fontname"])
        cbar = figure.colorbar(image, ax=axis, shrink=0.8)
        cbar.set_label("改进幅度(微调-原始)", fontsize=10, fontname=font_props["fontname"])
        plt_module.tight_layout()
        plt_module.show()

    @staticmethod
    def plot_iou_distribution(plt_module, all_results: Dict, title: str = "IOU分布对比") -> None:
        font_props = MetricsVisualizer._ensure_chinese_font(plt_module)
        splits = list(all_results.keys())
        figure, axis = plt_module.subplots(figsize=(10, 5))

        data_for_box = []
        labels_for_box = []
        positions = []
        position = 1

        for split_name in splits:
            split_results = all_results[split_name]
            base_ious = [result["base_metrics"]["mean_match_iou"] for result in split_results if result["base_metrics"]["mean_match_iou"] > 0]
            ft_ious = [result["ft_metrics"]["mean_match_iou"] for result in split_results if result["ft_metrics"]["mean_match_iou"] > 0]

            data_for_box.append(base_ious if base_ious else [0])
            labels_for_box.append(f"{split_name}-原始")
            positions.append(position)
            position += 1

            data_for_box.append(ft_ious if ft_ious else [0])
            labels_for_box.append(f"{split_name}-微调")
            positions.append(position)
            position += 2

        boxplot = axis.boxplot(data_for_box, positions=positions, patch_artist=True, widths=0.6)
        for index, patch in enumerate(boxplot["boxes"]):
            model_type = "原始" if "原始" in labels_for_box[index] else "微调"
            color_key = "原始模型" if model_type == "原始" else "微调模型"
            patch.set_facecolor(MetricsVisualizer.MODEL_COLORS[color_key])
            patch.set_alpha(0.7)

        axis.set_xticklabels(labels_for_box, fontsize=9, fontname=font_props["fontname"])
        axis.set_ylabel("匹配IOU", fontsize=11, fontname=font_props["fontname"])
        axis.set_title(title, fontsize=13, fontweight="bold", fontname=font_props["fontname"])
        axis.grid(axis="y", alpha=0.3)
        plt_module.tight_layout()
        plt_module.show()


class SequentialEvaluator:
    def __init__(
        self,
        iou_threshold: float = 0.5,
        batch_size: int = 1,
        max_new_tokens: int = 512,
        result_dir: Optional[str] = None,
        use_cache: bool = True,
        progress_callback: Optional[callable] = None,
    ):
        self.iou_threshold = iou_threshold
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.result_dir = result_dir
        self.use_cache = use_cache
        self.progress_callback = progress_callback
        self.result_manager = ResultManager(result_dir) if result_dir else None

    def _prepare_records(self, records: List[Dict], max_samples: Optional[int] = None) -> List[Dict]:
        eval_records = records[:max_samples] if max_samples and max_samples < len(records) else records
        prepared = []
        for record in eval_records:
            image_path = DatasetLoader.extract_image_path(record)
            image = DatasetLoader.load_image(image_path)
            if image is None:
                continue
            prepared.append(
                {
                    "image": image,
                    "image_path": image_path,
                    "query": DatasetLoader.extract_query(record),
                    "ground_truth": DatasetLoader.parse_ground_truth(record),
                    "original_record": record,
                }
            )
        return prepared

    def _run_batch_inference(
        self,
        detector: "ObjectDetector",
        records: List[Dict],
        model_key: str,
        split_name: str,
    ) -> List[Dict]:
        results = []
        total = len(records)

        if self.batch_size <= 1:
            for index, record in enumerate(records):
                if self.progress_callback:
                    self.progress_callback(index, total, model_key, split_name)
                det_result = detector.detect(
                    record["image"],
                    record["query"],
                    max_new_tokens=self.max_new_tokens,
                )
                detections = det_result.get("detections", []) if det_result.get("success") else []
                metrics = MetricsCalculator.compute_sample_metrics(detections, record["ground_truth"], self.iou_threshold)
                results.append(
                    {
                        "index": index,
                        "image_path": record["image_path"],
                        "query": record["query"],
                        "detections": detections,
                        "ground_truth": record["ground_truth"],
                        "metrics": metrics,
                    }
                )
        else:
            for batch_start in range(0, total, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total)
                batch_records = records[batch_start:batch_end]

                if self.progress_callback:
                    self.progress_callback(batch_start, total, model_key, split_name, batch_size=len(batch_records))

                images = [r["image"] for r in batch_records]
                queries = [r["query"] for r in batch_records]

                batch_results = detector.detect_batch(images, queries, max_new_tokens=self.max_new_tokens, batch_size=len(images))

                for idx, (record, det_result) in enumerate(zip(batch_records, batch_results)):
                    detections = det_result.get("detections", []) if det_result.get("success") else []
                    metrics = MetricsCalculator.compute_sample_metrics(detections, record["ground_truth"], self.iou_threshold)
                    results.append(
                        {
                            "index": batch_start + idx,
                            "image_path": record["image_path"],
                            "query": record["query"],
                            "detections": detections,
                            "ground_truth": record["ground_truth"],
                            "metrics": metrics,
                        }
                    )

        return results

    def evaluate_single_model(
        self,
        model_loader: "ModelLoader",
        records: List[Dict],
        model_key: str,
        split_name: str,
        coord_format: str = "xyxy",
        coord_norm: str = "auto",
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_samples: Optional[int] = None,
        detector: Optional["ObjectDetector"] = None,
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        prepared_records = self._prepare_records(records, max_samples)

        if self.use_cache and self.result_manager:
            cached_samples = self.result_manager.load_sample_results(model_key, split_name)
            cached_agg = self.result_manager.load_aggregated_metrics(model_key, split_name)
            if cached_samples and cached_agg:
                print(f"[{model_key}/{split_name}] 使用缓存结果 ({len(cached_samples)} 条)")
                return cached_samples, cached_agg

        own_detector = False
        if detector is None:
            from unsloth_finetune.notebooking.vision_shared import ObjectDetector

            own_detector = True
            if not model_loader.is_loaded():
                print(f"[{model_key}] 正在加载模型...")
                load_success = model_loader.load_model()
                if not load_success:
                    raise RuntimeError(f"[{model_key}] 模型加载失败，无法进行推理")
            detector = ObjectDetector(
                model_loader,
                temperature=temperature,
                top_p=top_p,
                coord_format=coord_format,
                coord_norm=coord_norm,
            )

        results = self._run_batch_inference(detector, prepared_records, model_key, split_name)

        metrics_list = [r["metrics"] for r in results]
        aggregated = aggregate_metric_dicts(metrics_list)

        if self.use_cache and self.result_manager:
            self.result_manager.save_sample_results(model_key, split_name, results)
            self.result_manager.save_aggregated_metrics(model_key, split_name, aggregated)

        if own_detector:
            pass

        return results, aggregated

    def evaluate_all(
        self,
        datasets: Dict[str, List[Dict]],
        model_loader_finetuned: "ModelLoader",
        model_loader_base: "ModelLoader",
        coord_format: str = "xyxy",
        coord_norm: str = "auto",
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_samples: Optional[int] = None,
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict[str, Dict[str, Any]]]]:
        all_results: Dict[str, List[Dict]] = {}
        all_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}

        print(f"\n{'=' * 60}")
        print(f"SequentialEvaluator 开始评估 (batch_size={self.batch_size})")
        print(f"{'=' * 60}\n")

        for split_name, records in datasets.items():
            if not records:
                print(f"跳过空数据集: {split_name}")
                continue

            all_results[split_name] = []
            all_metrics[split_name] = {}

            print(f"\n[{split_name}] 评估开始 ({len(records)} 条记录)")

            print(f"\n  Phase 1: 加载微调模型进行推理...")
            ft_results, ft_agg = self.evaluate_single_model(
                model_loader=model_loader_finetuned,
                records=records,
                model_key="finetuned",
                split_name=split_name,
                coord_format=coord_format,
                coord_norm=coord_norm,
                temperature=temperature,
                top_p=top_p,
                max_samples=max_samples,
            )
            all_results[split_name].extend(
                [
                    {
                        "index": r["index"],
                        "split": split_name,
                        "image_path": r["image_path"],
                        "query": r["query"],
                        "ground_truth": r["ground_truth"],
                        "det_ft": r["detections"],
                        "ft_metrics": r["metrics"],
                    }
                    for r in ft_results
                ]
            )
            all_metrics[split_name]["finetuned"] = ft_agg
            print(f"  微调模型评估完成: F1={ft_agg.get('mean_f1', 0):.3f}")

            print(f"\n  Phase 2: 卸载微调模型, 加载基础模型进行推理...")
            model_loader_finetuned.unload_model()

            base_results, base_agg = self.evaluate_single_model(
                model_loader=model_loader_base,
                records=records,
                model_key="base",
                split_name=split_name,
                coord_format=coord_format,
                coord_norm=coord_norm,
                temperature=temperature,
                top_p=top_p,
                max_samples=max_samples,
            )

            for idx, r in enumerate(base_results):
                if idx < len(all_results[split_name]):
                    all_results[split_name][idx]["det_base"] = r["detections"]
                    all_results[split_name][idx]["base_metrics"] = r["metrics"]
                    all_results[split_name][idx]["model_iou"] = IOUCalculator.calculate_batch_iou(r["detections"], all_results[split_name][idx]["det_ft"])

            all_metrics[split_name]["base"] = base_agg
            print(f"  基础模型评估完成: F1={base_agg.get('mean_f1', 0):.3f}")

            model_loader_base.unload_model()
            print(f"\n[{split_name}] 评估完成")

        print(f"\n{'=' * 60}")
        print(f"所有数据集评估完成!")
        print(f"{'=' * 60}\n")

        return all_results, all_metrics


class BatchEvaluator:
    def __init__(
        self,
        detector_base: "ObjectDetector",
        detector_finetuned: "ObjectDetector",
        iou_threshold: float = 0.5,
        batch_size: int = 1,
        max_new_tokens: int = 512,
        progress_callback: Optional[callable] = None,
    ):
        self.detector_base = detector_base
        self.detector_finetuned = detector_finetuned
        self.iou_threshold = iou_threshold
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.progress_callback = progress_callback

    def evaluate_dataset(
        self,
        records: List[Dict],
        split_name: str,
        max_samples: Optional[int] = None,
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        eval_records = records[:max_samples] if max_samples and max_samples < len(records) else records

        sample_results = []
        total = len(eval_records)

        if self.batch_size <= 1:
            for index, record in enumerate(eval_records):
                if self.progress_callback:
                    self.progress_callback(index, total, split_name)

                image_path = DatasetLoader.extract_image_path(record)
                image = DatasetLoader.load_image(image_path)
                if image is None:
                    continue

                query = DatasetLoader.extract_query(record)
                ground_truth = DatasetLoader.parse_ground_truth(record)

                det_base_result = self.detector_base.detect(image, query, max_new_tokens=self.max_new_tokens)
                det_ft_result = self.detector_finetuned.detect(image, query, max_new_tokens=self.max_new_tokens)

                det_base = det_base_result.get("detections", []) if det_base_result.get("success") else []
                det_ft = det_ft_result.get("detections", []) if det_ft_result.get("success") else []

                base_metrics = MetricsCalculator.compute_sample_metrics(det_base, ground_truth, self.iou_threshold)
                ft_metrics = MetricsCalculator.compute_sample_metrics(det_ft, ground_truth, self.iou_threshold)
                model_iou = IOUCalculator.calculate_batch_iou(det_base, det_ft)

                sample_results.append(
                    {
                        "index": index,
                        "split": split_name,
                        "image_path": image_path,
                        "query": query,
                        "ground_truth": ground_truth,
                        "det_base": det_base,
                        "det_ft": det_ft,
                        "base_metrics": base_metrics,
                        "ft_metrics": ft_metrics,
                        "model_iou": model_iou,
                    }
                )
        else:
            for batch_start in range(0, total, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total)
                batch_records = eval_records[batch_start:batch_end]

                if self.progress_callback:
                    self.progress_callback(batch_start, total, split_name, batch_size=len(batch_records))

                images = []
                queries = []
                valid_indices = []

                for idx, record in enumerate(batch_records):
                    image_path = DatasetLoader.extract_image_path(record)
                    image = DatasetLoader.load_image(image_path)
                    if image is None:
                        continue
                    images.append(image)
                    queries.append(DatasetLoader.extract_query(record))
                    valid_indices.append(batch_start + idx)

                if not images:
                    continue

                base_results = self.detector_base.detect_batch(images, queries, max_new_tokens=self.max_new_tokens, batch_size=len(images))
                ft_results = self.detector_finetuned.detect_batch(images, queries, max_new_tokens=self.max_new_tokens, batch_size=len(images))

                for idx, (image, query, base_res, ft_res) in enumerate(zip(images, queries, base_results, ft_results)):
                    record_idx = valid_indices[idx]
                    record = batch_records[idx]

                    ground_truth = DatasetLoader.parse_ground_truth(record)
                    det_base = base_res.get("detections", []) if base_res.get("success") else []
                    det_ft = ft_res.get("detections", []) if ft_res.get("success") else []

                    base_metrics = MetricsCalculator.compute_sample_metrics(det_base, ground_truth, self.iou_threshold)
                    ft_metrics = MetricsCalculator.compute_sample_metrics(det_ft, ground_truth, self.iou_threshold)
                    model_iou = IOUCalculator.calculate_batch_iou(det_base, det_ft)

                    sample_results.append(
                        {
                            "index": record_idx,
                            "split": split_name,
                            "image_path": DatasetLoader.extract_image_path(record),
                            "query": query,
                            "ground_truth": ground_truth,
                            "det_base": det_base,
                            "det_ft": det_ft,
                            "base_metrics": base_metrics,
                            "ft_metrics": ft_metrics,
                            "model_iou": model_iou,
                        }
                    )

        base_metric_list = [r["base_metrics"] for r in sample_results]
        ft_metric_list = [r["ft_metrics"] for r in sample_results]

        aggregated = {
            "base": aggregate_metric_dicts(base_metric_list),
            "finetuned": aggregate_metric_dicts(ft_metric_list),
        }

        return sample_results, aggregated
