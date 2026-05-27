#!/usr/bin/env python3
# Unsloth 必须在所有其他导入之前导入以确保优化生效
import unsloth  # noqa: F401

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.distributed as dist
from PIL import Image
from unsloth_finetune.data.labelme.progress_logger import TQDM_AVAILABLE, create_progress_bar
from unsloth_finetune.core.labelme_export import save_labelme_results
from unsloth_finetune.core.runtime import (
    configure_unsloth_compile_cache,
    format_log_timestamp,
    get_env_value,
    resolve_notebook_dir,
)
from unsloth_finetune.training.distributed.load_balancer import (
    SQLiteTaskQueue,
    build_observed_scheduler_report,
    build_scheduler_comparison,
    collect_local_gpu_metrics,
    render_scheduler_report_markdown,
)
from unsloth_finetune.training.distributed.adapter_utils import prepared_adapter_dir
from unsloth_finetune.data.labelme.detection_format import (
    DetectionPromptBuilder,
    build_cn_detection_prompt,
    build_en_detection_prompt,
    convert_xyxy_to_format,
)

NOTEBOOK_DIR = resolve_notebook_dir(
    cwd=Path.cwd(),
    notebook_file=get_env_value("UNSLOTH_NOTEBOOK_FILE", "GEMMA4_NOTEBOOK_FILE"),
)
UNSLOTH_CACHE_DIR = configure_unsloth_compile_cache(NOTEBOOK_DIR)

from unsloth import FastVisionModel

INFERENCE_TEMPERATURE = 0.7
INFERENCE_TOP_P = 0.9
INFERENCE_MAX_NEW_TOKENS = 512

_VERBOSE_STATUS_CACHE: Optional[bool] = None
_LIVE_TQDM_RANKS_CACHE: Optional[Optional[set]] = None
_FORCE_LIVE_TQDM_CACHE: Optional[bool] = None


def verbose_status_enabled() -> bool:
    global _VERBOSE_STATUS_CACHE
    if _VERBOSE_STATUS_CACHE is None:
        _VERBOSE_STATUS_CACHE = get_env_value("UNSLOTH_VERBOSE_TQDM_STATUS", "GEMMA4_VERBOSE_TQDM_STATUS") == "1"
    return _VERBOSE_STATUS_CACHE


def parse_live_tqdm_ranks() -> Optional[set]:
    global _LIVE_TQDM_RANKS_CACHE
    if _LIVE_TQDM_RANKS_CACHE is not None:
        return _LIVE_TQDM_RANKS_CACHE
    raw = get_env_value("UNSLOTH_LIVE_TQDM_RANKS", "GEMMA4_LIVE_TQDM_RANKS").strip().lower() or "none"
    if not raw or raw in {"none", "off", "disabled"}:
        _LIVE_TQDM_RANKS_CACHE = set()
        return set()
    if raw in {"all", "*"}:
        _LIVE_TQDM_RANKS_CACHE = None
        return None

    ranks = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ranks.add(int(chunk))
        except ValueError:
            continue
    _LIVE_TQDM_RANKS_CACHE = ranks
    return ranks


def live_tqdm_enabled(rank: Optional[int] = None) -> bool:
    global _FORCE_LIVE_TQDM_CACHE
    if _FORCE_LIVE_TQDM_CACHE is None:
        override = get_env_value("UNSLOTH_FORCE_LIVE_TQDM", "GEMMA4_FORCE_LIVE_TQDM").strip().lower()
        if override in {"1", "true", "yes", "on"}:
            _FORCE_LIVE_TQDM_CACHE = True
        elif override in {"0", "false", "no", "off"}:
            _FORCE_LIVE_TQDM_CACHE = False
        else:
            _FORCE_LIVE_TQDM_CACHE = None
    if _FORCE_LIVE_TQDM_CACHE is True:
        return True
    if _FORCE_LIVE_TQDM_CACHE is False:
        return False
    if not sys.stderr.isatty():
        return False

    enabled_ranks = parse_live_tqdm_ranks()
    if enabled_ranks is None or rank is None:
        return True
    return rank in enabled_ranks


@contextmanager
def quiet_library_output():
    if verbose_status_enabled():
        yield
        return

    with ExitStack() as stack:
        stack.enter_context(redirect_stdout(StringIO()))
        stack.enter_context(redirect_stderr(StringIO()))
        yield


def normalize_device_spec(device_value: Any) -> str:
    if isinstance(device_value, torch.device):
        return str(device_value)

    text = str(device_value).strip()
    if not text:
        return "unknown"
    if text.isdigit():
        return f"cuda:{int(text)}"
    return text


def extract_model_device_targets(model: Any) -> List[str]:
    candidates = [model, getattr(model, "base_model", None), getattr(model, "model", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        device_map = getattr(candidate, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            return sorted({normalize_device_spec(value) for value in device_map.values()})

    try:
        return [normalize_device_spec(next(model.parameters()).device)]
    except (AttributeError, StopIteration, TypeError):
        return []


def write_console(message: Any) -> None:
    text = str(message).rstrip("\r\n")
    if not text:
        return

    if TQDM_AVAILABLE:
        try:
            from tqdm import tqdm as std_tqdm

            std_tqdm.write(text, file=sys.stderr)
            return
        except Exception:
            pass

    sys.stderr.write(text + "\n")
    sys.stderr.flush()


def write_verbose_console(message: Any) -> None:
    if verbose_status_enabled():
        write_console(message)


def now_tag() -> str:
    return format_log_timestamp()


def log(message: str, rank: Optional[int] = None) -> None:
    prefix = f"[rank {rank}] " if rank is not None else ""
    write_console(f"[{now_tag()}] {prefix}{message}")


def log_exception(message: str, exc: Exception, rank: Optional[int] = None) -> None:
    log(f"{message}: {exc}", rank=rank)
    for line in traceback.format_exc().strip().splitlines():
        log(line, rank=rank)


def format_progress_summary(
    model_type: str,
    *,
    total: int,
    completed: int,
    processed: int,
    failed: int,
    remaining: int,
) -> str:
    width = max(3, len(str(max(total, completed, processed, failed, remaining, 0))))
    model_label = f"{model_type:<9}"
    return f"{model_label} 数据集共 {total:>{width}} 条，" f"当前已处理 {completed:>{width}} 条，" f"成功 {processed:>{width}} 条，" f"失败 {failed:>{width}} 条，" f"剩余 {remaining:>{width}} 条"


def create_inference_progress_bar(
    *,
    total: int,
    model_type: str,
    rank: int,
    physical_gpu: int,
    dynamic_mode: bool = False,
):
    desc = f"rank {rank} gpu {physical_gpu} {model_type}"
    if dynamic_mode:
        desc += " overall"
    return create_progress_bar(
        total=total,
        desc=desc,
        unit="条",
        position=rank,
        smoothing=0.1,
        bar_format="{desc}: {percentage:6.2f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )


def sync_progress_bar(progress_bar: Any, completed: int, *, processed: int, failed: int) -> None:
    if progress_bar is None:
        return

    target = max(0, int(completed))
    delta = target - int(getattr(progress_bar, "n", 0))
    if delta > 0:
        progress_bar.update(delta)
    progress_bar.set_postfix(processed=processed, failed=failed, refresh=False)


def chunked(items: List[Any], size: int):
    size = max(1, int(size or 1))
    for start in range(0, len(items), size):
        yield items[start : start + size]


class IOUCalculator:
    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
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


class MetricsCalculator:
    @staticmethod
    def _normalize_label(label: Any) -> str:
        return str(label or "").strip().lower()

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
                "precision": 0.0,
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

        for i, det in enumerate(detections):
            det_bbox = det.get("bbox", [0, 0, 0, 0])
            det_label = MetricsCalculator._normalize_label(det.get("label"))
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(ground_truth):
                if j in matched_gt:
                    continue
                gt_label = MetricsCalculator._normalize_label(gt.get("label"))
                if det_label != gt_label:
                    continue
                gt_bbox = gt.get("bbox", [0, 0, 0, 0])
                iou = IOUCalculator.calculate_iou(det_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_threshold and best_j >= 0:
                matched_gt.add(best_j)
                matched_det.add(i)
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
            values = [m[key] for m in sample_metrics_list if key in m]
            result[f"mean_{key}"] = float(np.mean(values)) if values else 0.0
            result[f"std_{key}"] = float(np.std(values)) if values else 0.0

        result["total_samples"] = len(sample_metrics_list)
        result["success_rate"] = float(np.mean([m.get("det_success", False) for m in sample_metrics_list]))
        return result


class DatasetLoader:
    @staticmethod
    def load_jsonl(filepath: str) -> List[Dict]:
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
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
        metadata = record.get("metadata", {})
        img_width = metadata.get("image_width", 1000)
        img_height = metadata.get("image_height", 1000)

        messages = record.get("messages", [])
        assistant_text = ""
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    assistant_text += item.get("text", "")

        if not assistant_text:
            return []

        # Try box_2d_json format if metadata indicates it or text starts with '['
        output_format = metadata.get("output_format", "labelme_text")
        if output_format == "box_2d_json" or assistant_text.lstrip().startswith("["):
            try:
                box_2d_results = parse_box_2d_json_ground_truth(assistant_text, img_width, img_height)
                if box_2d_results:
                    return box_2d_results
            except Exception:
                pass  # Fall through to legacy regex parsing

        pattern = r"-\s*(\S+)\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
        gt_bboxes = []
        for match in re.finditer(pattern, assistant_text):
            label = match.group(1)
            coords = [float(match.group(i)) for i in range(2, 6)]
            x_min, y_min, x_max, y_max = coords

            if all(0 <= c <= 1 for c in coords):
                x_min_px = int(x_min * img_width)
                y_min_px = int(y_min * img_height)
                x_max_px = int(x_max * img_width)
                y_max_px = int(y_max * img_height)
            else:
                x_min_px = int(x_min)
                y_min_px = int(y_min)
                x_max_px = int(x_max)
                y_max_px = int(y_max)

            gt_bboxes.append(
                {
                    "bbox": [x_min_px, y_min_px, x_max_px, y_max_px],
                    "label": label,
                    "confidence": 1.0,
                }
            )

        return gt_bboxes

    @staticmethod
    def extract_query(record: Dict) -> str:
        for msg in record.get("messages", []):
            if msg.get("role") != "user":
                continue
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    return item.get("text", "")
        return ""

    @staticmethod
    def extract_image_path(record: Dict) -> str:
        images = record.get("images", [])
        if images:
            return images[0]
        metadata = record.get("metadata", {})
        return metadata.get("json_path", "")

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


class ModelLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
        self._is_loaded = False

    def _patch_peft_for_gemma4(self) -> bool:
        try:
            from peft.tuners.lora import model as lora_model

            original = lora_model.LoraModel._create_new_module

            def patched(lora_config, adapter_name, target, **kwargs):
                if target.__class__.__name__ == "Gemma4ClippableLinear" and hasattr(target, "linear"):
                    return original(lora_config, adapter_name, target.linear, **kwargs)
                return original(lora_config, adapter_name, target, **kwargs)

            lora_model.LoraModel._create_new_module = staticmethod(patched)
            write_verbose_console("PEFT patched for Gemma4ClippableLinear")
            return True
        except Exception as exc:
            write_console(f"PEFT patch failed: {exc}")
            return False

    def _validate_device_placement(self) -> None:
        expected_device = str(self.config.get("expected_device", "")).strip()
        if not expected_device or expected_device == "cpu":
            return

        targets = extract_model_device_targets(self.model)
        if not targets:
            return

        unexpected = [device for device in targets if device != expected_device]
        if unexpected:
            raise RuntimeError(f"model_device_mismatch: expected={expected_device}, actual={','.join(targets)}")

    def load_model(self) -> bool:
        try:
            write_verbose_console(f"Loading model: {self.config.get('name', 'Unknown')}")
            os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
            if not self.config.get("enable_compile", False):
                os.environ["TORCH_COMPILE_DISABLE"] = "1"
                try:
                    import torch._dynamo

                    torch._dynamo.config.suppress_errors = True
                    torch._dynamo.reset()
                except Exception:
                    pass

            from_pretrained_kwargs = {
                "model_name": self.config["base_model_path"],
                "max_seq_length": self.config["max_seq_length"],
                "load_in_4bit": self.config["load_in_4bit"],
                "device_map": self.config["device_map"],
                "disable_log_stats": True,
            }
            attn_impl = self.config.get("attn_implementation")
            if attn_impl is not None:
                from_pretrained_kwargs["attn_implementation"] = attn_impl

            with quiet_library_output():
                self.model, self.processor = FastVisionModel.from_pretrained(**from_pretrained_kwargs)

                # 确认注意力实现: 检查模型实际使用的attn kernel
                _resolved_attn = getattr(self.model.config, "_attn_implementation", None) or getattr(self.model.config, "attn_implementation", None)
                _requested_attn = self.config.get("attn_implementation")
                log(f"注意力实现: resolved={_resolved_attn}, requested={_requested_attn}")

                lora_path = self.config.get("lora_adapter_path")
                if lora_path and os.path.exists(lora_path):
                    write_verbose_console(f"Loading LoRA adapter: {lora_path}")
                    self._patch_peft_for_gemma4()
                    from peft import PeftModel

                    with prepared_adapter_dir(lora_path) as prepared_lora_path:
                        self.model = PeftModel.from_pretrained(
                            self.model,
                            str(prepared_lora_path),
                            is_trainable=False,
                        )
                    write_verbose_console("LoRA adapter loaded")

            self._validate_device_placement()

            self._is_loaded = True
            return True
        except Exception as exc:
            write_console(f"Model load failed: {exc}")
            import traceback

            traceback.print_exc()
            return False

    def unload_model(self) -> None:
        if not self._is_loaded:
            return

        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def is_loaded(self) -> bool:
        return self._is_loaded


class ObjectDetector:
    def __init__(self, model_loader: ModelLoader, prompt_builder: Optional[DetectionPromptBuilder] = None, coord_format: str = "xyxy"):
        self.model_loader = model_loader
        self.prompt_builder = prompt_builder
        self.coord_format = coord_format

    def _build_prompt(self, query: str) -> str:
        query_text = str(query or "").strip()
        if not query_text:
            return query_text

        # 训练数据里的 query 已经是完整自然语言指令时，评估直接复用，
        # 避免再套一层模板造成口径漂移。
        if (
            "识别并定位其中的" in query_text
            or query_text.startswith("请检测图片中")
            or query_text.startswith("请检测图片")
            or query_text.startswith("Please detect")
            or query_text.startswith("Detect")
        ):
            return query_text

        if self.prompt_builder:
            return self.prompt_builder(query_text)
        return build_en_detection_prompt(query_text)

    def _resolve_model_device(self, model) -> torch.device:
        model_device = getattr(model, "device", None)
        if model_device is not None:
            return model_device
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _move_inputs_to_device(self, inputs, device: torch.device):
        if hasattr(inputs, "to"):
            return inputs.to(device)
        if isinstance(inputs, dict):
            moved = {}
            for key, value in inputs.items():
                moved[key] = value.to(device) if hasattr(value, "to") else value
            return moved
        raise TypeError(f"Unsupported processor output type: {type(inputs)!r}")

    def _prepare_generation_inputs(self, images: List[Image.Image], queries: List[str], padding: bool):
        model = self.model_loader.model
        processor = self.model_loader.processor
        prompts = [self._build_prompt(query) for query in queries]
        messages_batch = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for image, prompt in zip(images, prompts)
        ]
        texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
        processor_kwargs = {
            "text": texts,
            "images": images,
            "return_tensors": "pt",
        }
        if padding:
            processor_kwargs["padding"] = True
        inputs = processor(**processor_kwargs)
        return self._move_inputs_to_device(inputs, self._resolve_model_device(model))

    def _decode_generated_responses(self, processor, inputs, outputs) -> List[str]:
        if "attention_mask" in inputs:
            prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        elif "input_ids" in inputs:
            prompt_lengths = [inputs["input_ids"].shape[1]] * inputs["input_ids"].shape[0]
        else:
            prompt_lengths = [0] * outputs.shape[0]

        generated_only = []
        for idx, output in enumerate(outputs):
            start = int(prompt_lengths[idx]) if idx < len(prompt_lengths) else 0
            generated_only.append(output[start:])
        return processor.batch_decode(generated_only, skip_special_tokens=True)

    def detect(self, image: Image.Image, query: str, max_new_tokens: int = 512) -> Dict[str, Any]:
        if not self.model_loader.is_loaded():
            return {"error": "model not loaded", "success": False}

        try:
            model = self.model_loader.model
            processor = self.model_loader.processor
            inputs = self._prepare_generation_inputs([image], [query], padding=False)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    temperature=INFERENCE_TEMPERATURE,
                    top_p=INFERENCE_TOP_P,
                    do_sample=True,
                )

            response = self._decode_generated_responses(processor, inputs, outputs)[0]
            width, height = image.size
            detections = self._parse_response(response, width, height, coord_format=self.coord_format)
            return {"success": True, "raw_response": response, "detections": detections, "query": query}
        except Exception as exc:
            return {"error": str(exc), "success": False}

    def detect_batch(
        self,
        images: List[Image.Image],
        queries: List[str],
        max_new_tokens: int = 512,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.model_loader.is_loaded():
            return [{"error": "model not loaded", "success": False} for _ in images]
        if not images:
            return []
        if len(images) != len(queries):
            raise ValueError(f"images and queries length mismatch: {len(images)} != {len(queries)}")

        model = self.model_loader.model
        processor = self.model_loader.processor
        batch_size = batch_size or len(images)
        results: List[Dict[str, Any]] = []
        for start in range(0, len(images), batch_size):
            image_batch = images[start : start + batch_size]
            query_batch = queries[start : start + batch_size]
            try:
                inputs = self._prepare_generation_inputs(image_batch, query_batch, padding=len(image_batch) > 1)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        temperature=INFERENCE_TEMPERATURE,
                        top_p=INFERENCE_TOP_P,
                        do_sample=True,
                    )

                responses = self._decode_generated_responses(processor, inputs, outputs)
                for image, query, response in zip(image_batch, query_batch, responses):
                    width, height = image.size
                    detections = self._parse_response(response, width, height, coord_format=self.coord_format)
                    results.append(
                        {
                            "success": True,
                            "raw_response": response,
                            "detections": detections,
                            "query": query,
                        }
                    )
            except Exception as batch_error:
                for image, query in zip(image_batch, query_batch):
                    single = self.detect(image, query, max_new_tokens=max_new_tokens)
                    if not single.get("success"):
                        single["error"] = single.get("error") or str(batch_error)
                    results.append(single)
        return results

    def _parse_response(self, response: str, width: int, height: int, coord_format: str = "xyxy") -> List[Dict[str, Any]]:
        detections = []

        def is_normalized(coords: list) -> bool:
            return all(0 <= v <= 1 for v in coords)

        def convert_coords(coords: list) -> Tuple[int, int, int, int]:
            if is_normalized(coords):
                x1 = int(coords[0] * width)
                y1 = int(coords[1] * height)
                x2 = int(coords[2] * width)
                y2 = int(coords[3] * height)
            else:
                x1 = int(coords[0])
                y1 = int(coords[1])
                x2 = int(coords[2])
                y2 = int(coords[3])
            return x1, y1, x2, y2

        def sanitize_box(x1: int, y1: int, x2: int, y2: int):
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            if x2 <= x1 or y2 <= y1:
                return None
            return x1, y1, x2, y2

        def extract_json_array(text: str) -> Optional[str]:
            json_block = re.search(r"```json\s*([\s\S]*?)\s*```", text)
            if json_block:
                return json_block.group(1).strip()
            start_idx = text.find("[")
            if start_idx == -1:
                return None
            try:
                parsed, end_idx = json.JSONDecoder().raw_decode(text[start_idx:])
                return text[start_idx : start_idx + end_idx]
            except json.JSONDecodeError:
                pass
            bracket_count = 0
            for i, char in enumerate(text[start_idx:], start_idx):
                if char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        return text[start_idx : i + 1]
            return None

        def append_detection(item: dict):
            coords = item.get("box_2d")
            if not isinstance(coords, list) or len(coords) != 4:
                return
            x1, y1, x2, y2 = convert_coords(coords)
            sanitized = sanitize_box(x1, y1, x2, y2)
            if sanitized is None:
                return
            confidence = item.get("confidence", 0.85)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.85
            detections.append(
                {
                    "bbox": [sanitized[0], sanitized[1], sanitized[2], sanitized[3]],
                    "bbox_out": convert_xyxy_to_format(sanitized[0], sanitized[1], sanitized[2], sanitized[3], coord_format),
                    "label": item.get("label", "object"),
                    "confidence": max(0.0, min(confidence, 1.0)),
                }
            )

        json_str = extract_json_array(response)
        if json_str:
            try:
                json_data = json.loads(json_str)
                if isinstance(json_data, list):
                    for item in json_data:
                        if isinstance(item, dict):
                            append_detection(item)
                if detections:
                    return detections
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        obj_pattern = r'\{[^{}]*"box_2d"[^{}]*\}'
        for obj_str in re.findall(obj_pattern, response, re.DOTALL):
            try:
                obj = json.loads(obj_str)
                if isinstance(obj, dict):
                    append_detection(obj)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return detections


def setup_distributed(gpu_ids: List[int]):
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        os.environ["NCCL_P2P_LEVEL"] = os.environ.get("NCCL_P2P_LEVEL", "SYS")
        os.environ["NCCL_IB_DISABLE"] = os.environ.get("NCCL_IB_DISABLE", "1")
        init_kwargs: Dict[str, Any] = {"backend": backend}
        try:
            import inspect

            signature = inspect.signature(dist.init_process_group)
            if "device_id" in signature.parameters and torch.cuda.is_available():
                init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        except Exception:
            pass
        dist.init_process_group(**init_kwargs)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    physical_gpu = gpu_ids[local_rank] if local_rank < len(gpu_ids) else local_rank
    return rank, local_rank, world_size, is_distributed, physical_gpu


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_partition(
    records: List[Dict],
    rank: int,
    world_size: int,
    strategy: str = "contiguous",
) -> List[Dict]:
    if strategy == "round_robin":
        return records[rank::world_size]

    total = len(records)
    base = total // world_size
    remainder = total % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return records[start:end]


def ensure_record_indices(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    indexed_records: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        if record.get("index") == index:
            indexed_records.append(record)
            continue
        indexed = dict(record)
        indexed["index"] = index
        indexed_records.append(indexed)
    return indexed_records


def estimate_record_complexity(record: Dict[str, Any]) -> float:
    """A cheap heuristic used to prioritize expensive-looking samples in the queue."""
    score = 1.0
    try:
        image_path = DatasetLoader.extract_image_path(record)
        image_file = Path(image_path)
        if image_file.exists():
            score += min(image_file.stat().st_size / (1024**2), 20.0)
    except Exception:
        pass

    try:
        query = DatasetLoader.extract_query(record)
        score += min(len(query) / 128.0, 4.0)
    except Exception:
        pass

    try:
        score += min(len(DatasetLoader.parse_ground_truth(record)) * 0.5, 4.0)
    except Exception:
        pass

    return round(score, 3)


def build_dynamic_task_payloads(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payloads = []
    for record in records:
        payloads.append(
            {
                "sample_index": int(record.get("index", 0)),
                "complexity_score": estimate_record_complexity(record),
                "image_path": DatasetLoader.extract_image_path(record),
                "query_text": DatasetLoader.extract_query(record),
            }
        )
    return payloads


def get_queue_db_path(result_dir: Path, model_type: str) -> Path:
    return result_dir / "queues" / f"{model_type}_tasks.sqlite"


def build_worker_rows_from_stats(stats_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for stats in stats_map.values():
        rows.append(
            {
                "worker_rank": int(stats.get("rank", 0)),
                "gpu_id": int(stats.get("gpu_id", stats.get("rank", 0))),
                "state": stats.get("state", "done"),
                "claimed_samples": int(stats.get("claimed_samples", stats.get("processed", 0) + stats.get("failed", 0))),
                "processed": int(stats.get("processed", 0)),
                "failed": int(stats.get("failed", 0)),
                "total_inference_seconds": float(stats.get("inference_seconds", 0.0)),
                "utilization_pct": float(stats.get("utilization_pct", 0.0)),
                "memory_alloc_gb": float(stats.get("memory_alloc_gb", 0.0)),
                "memory_reserved_gb": float(stats.get("memory_reserved_gb", 0.0)),
                "temperature_c": float(stats.get("temperature_c", 0.0)),
                "queue_wait_seconds": float(stats.get("queue_wait_seconds", 0.0)),
                "last_batch_seconds": float(stats.get("last_batch_seconds", 0.0)),
            }
        )
    return sorted(rows, key=lambda item: item["worker_rank"])


def collect_sample_costs(results: List[Dict[str, Any]]) -> List[float]:
    costs = []
    for item in results:
        value = float(item.get("sample_wall_time_seconds", 0.0))
        if value > 0:
            costs.append(value)
    return costs


def save_load_balance_artifacts(
    *,
    result_dir: Path,
    model_type: str,
    scheduler_mode: str,
    partition_strategy: str,
    world_size: int,
    total_seconds: float,
    merged_results: List[Dict[str, Any]],
    worker_rows: List[Dict[str, Any]],
    task_counts: Optional[Dict[str, int]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    observed_report = build_observed_scheduler_report(
        worker_rows,
        total_seconds=total_seconds,
        world_size=world_size,
    )
    comparison = build_scheduler_comparison(
        sample_costs=collect_sample_costs(merged_results),
        world_size=world_size,
        scheduler_mode=scheduler_mode,
        static_partition_strategy=partition_strategy,
        observed_report=observed_report,
    )
    queue_report = {
        "scheduler_mode": scheduler_mode,
        "world_size": world_size,
        "task_counts": task_counts or {"pending": 0, "claimed": 0, "done": len(merged_results)},
        "worker_status": worker_rows,
        "observed": observed_report,
        "metadata": metadata or {},
    }

    report_dir = result_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{model_type}_load_balance_report.json"
    md_path = report_dir / f"{model_type}_load_balance_report.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": model_type,
                "queue_report": queue_report,
                "comparison": comparison,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(
            render_scheduler_report_markdown(
                model_type=model_type,
                comparison=comparison,
                queue_report=queue_report,
            )
        )

    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "queue_report": queue_report,
        "comparison": comparison,
    }


def build_model_config(args, model_type: str, local_rank: int, physical_gpu: int) -> Dict[str, Any]:
    expected_device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    config = {
        "name": f"{model_type}-GPU{physical_gpu}",
        "base_model_path": args.base_model_path,
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": args.load_in_4bit,
        "device_map": {"": local_rank} if torch.cuda.is_available() else "cpu",
        "expected_device": expected_device,
        "enable_compile": args.enable_compile,
    }
    if args.attn_implementation is not None:
        config["attn_implementation"] = args.attn_implementation
    if model_type == "finetuned" and args.lora_adapter_path:
        config["lora_adapter_path"] = args.lora_adapter_path
    return config


def validate_dynamic_queue_round(
    result_dir: Path,
    model_type: str,
    expected_total: int,
    world_size: int,
) -> Dict[str, int]:
    queue = SQLiteTaskQueue(get_queue_db_path(result_dir, model_type))
    return queue.validate_completion(
        expected_total=expected_total,
        expected_workers=world_size,
        require_done_workers=True,
    )


def wait_for_partial_results(
    result_dir: Path,
    model_type: str,
    *,
    expected_parts: int,
    expected_total: int,
    timeout_seconds: float = 30.0,
) -> Dict[str, int]:
    partial_dir = result_dir / "partials"
    deadline = time.time() + max(1.0, float(timeout_seconds))
    last_seen_parts = 0
    last_seen_results = 0

    while time.time() <= deadline:
        partial_paths = sorted(partial_dir.glob(f"{model_type}_rank*.json"))
        last_seen_parts = len(partial_paths)
        total_results = 0
        if last_seen_parts == expected_parts:
            for partial_path in partial_paths:
                with open(partial_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                total_results += len(payload.get("results", []))
            last_seen_results = total_results
            if total_results == expected_total:
                return {"partials": last_seen_parts, "results": total_results}
        time.sleep(0.2)

    raise RuntimeError(
        f"partial_results_incomplete: model_type={model_type}, expected_parts={expected_parts}, " f"seen_parts={last_seen_parts}, expected_results={expected_total}, seen_results={last_seen_results}"
    )


def save_partial_results(result_dir: Path, model_type: str, rank: int, payload: Dict[str, Any]) -> Path:
    partial_dir = result_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    output_path = partial_dir / f"{model_type}_rank{rank}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def run_model_round(
    model_type: str,
    partition_records: List[Dict],
    args,
    rank: int,
    local_rank: int,
    physical_gpu: int,
    prompt_builder: Optional[DetectionPromptBuilder] = None,
    coord_format: str = "xyxy",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    worker_start = time.time()
    load_start = time.time()
    log(f"{model_type} start loading model", rank=rank)
    loader = ModelLoader(build_model_config(args, model_type, local_rank, physical_gpu))
    success = loader.load_model()
    if success:
        log(f"{model_type} model loaded", rank=rank)
    load_seconds = time.time() - load_start

    if not success:
        return [], {
            "rank": rank,
            "gpu_id": physical_gpu,
            "claimed_samples": len(partition_records),
            "processed": 0,
            "failed": len(partition_records),
            "load_seconds": load_seconds,
            "inference_seconds": 0.0,
            "total_seconds": time.time() - worker_start,
            "throughput_samples_per_second": 0.0,
            "error": "model_load_failed",
            "scheduler_mode": args.scheduler_mode,
            "state": "failed",
        }

    detector = ObjectDetector(loader, prompt_builder=prompt_builder, coord_format=coord_format)
    results = []
    processed = 0
    failed = 0
    completed = 0
    infer_start = time.time()
    progress_bar = None
    last_logged_completed = -1

    if TQDM_AVAILABLE and partition_records and live_tqdm_enabled(rank):
        progress_bar = create_inference_progress_bar(
            total=len(partition_records),
            model_type=model_type,
            rank=rank,
            physical_gpu=physical_gpu,
        )

    try:
        for batch_records in chunked(partition_records, args.batch_size):
            batch_wall_start = time.time()
            prepared = []
            for record in batch_records:
                try:
                    image_path = DatasetLoader.extract_image_path(record)
                    query = DatasetLoader.extract_query(record)
                    ground_truth = DatasetLoader.parse_ground_truth(record)
                    image = DatasetLoader.load_image(image_path)
                    if image is None:
                        failed += 1
                        continue
                    prepared.append((record, image_path, query, ground_truth, image))
                except Exception as exc:
                    failed += 1
                    log(f"prepare sample failed on gpu {physical_gpu}: {exc}", rank=rank)

            if prepared:
                images = [item[4] for item in prepared]
                queries = [item[2] for item in prepared]
                det_results = detector.detect_batch(
                    images,
                    queries,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                )
                batch_elapsed = time.time() - batch_wall_start
                per_sample_elapsed = batch_elapsed / len(prepared) if prepared else 0.0

                for (record, image_path, query, ground_truth, _image), det_result in zip(prepared, det_results):
                    detections = det_result.get("detections", []) if det_result.get("success") else []
                    if not det_result.get("success", False):
                        failed += 1
                    metrics = MetricsCalculator.compute_sample_metrics(detections, ground_truth, args.iou_threshold)
                    results.append(
                        {
                            "gpu_id": physical_gpu,
                            "model_type": model_type,
                            "index": record.get("index", 0),
                            "image_path": image_path,
                            "image_width": _image.size[0],
                            "image_height": _image.size[1],
                            "query": query,
                            "detections": detections,
                            "metrics": metrics,
                            "raw_response": det_result.get("raw_response", "") if det_result.get("success") else "",
                            "inference_success": bool(det_result.get("success", False)),
                            "inference_error": det_result.get("error", "") if not det_result.get("success") else "",
                            "sample_wall_time_seconds": round(per_sample_elapsed, 6),
                        }
                    )
                    processed += 1

            completed += len(batch_records)
            if progress_bar is not None:
                sync_progress_bar(progress_bar, completed, processed=processed, failed=failed)
            if rank == 0 and completed != last_logged_completed:
                last_logged_completed = completed
                log(
                    format_progress_summary(
                        model_type,
                        total=len(partition_records),
                        completed=completed,
                        processed=processed,
                        failed=failed,
                        remaining=max(len(partition_records) - completed, 0),
                    ),
                    rank=rank,
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    inference_seconds = time.time() - infer_start
    gpu_metrics = collect_local_gpu_metrics(local_rank, physical_gpu)
    loader.unload_model()
    total_seconds = time.time() - worker_start
    stats = {
        "rank": rank,
        "gpu_id": physical_gpu,
        "claimed_samples": len(partition_records),
        "processed": processed,
        "failed": failed,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": total_seconds,
        "throughput_samples_per_second": processed / inference_seconds if inference_seconds > 0 else 0.0,
        "error": "",
        "scheduler_mode": args.scheduler_mode,
        "state": "done",
        **gpu_metrics,
    }
    return results, stats


def run_model_round_dynamic(
    model_type: str,
    all_records: List[Dict[str, Any]],
    args,
    rank: int,
    local_rank: int,
    physical_gpu: int,
    result_dir: Path,
    prompt_builder: Optional[DetectionPromptBuilder] = None,
    coord_format: str = "xyxy",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    queue_db_path = get_queue_db_path(result_dir, model_type)
    task_queue = SQLiteTaskQueue(queue_db_path)
    worker_start = time.time()
    load_start = time.time()
    log(f"{model_type} start loading model", rank=rank)
    loader = ModelLoader(build_model_config(args, model_type, local_rank, physical_gpu))
    success = loader.load_model()
    if success:
        log(f"{model_type} model loaded", rank=rank)
    load_seconds = time.time() - load_start
    queue_batch_size = max(1, int(args.queue_batch_size or args.batch_size))

    if not success:
        task_queue.update_worker_status(
            worker_rank=rank,
            gpu_id=physical_gpu,
            state="failed",
            claimed_samples=0,
            processed=0,
            failed=0,
            last_error="model_load_failed",
        )
        return [], {
            "rank": rank,
            "gpu_id": physical_gpu,
            "claimed_samples": 0,
            "processed": 0,
            "failed": 0,
            "load_seconds": load_seconds,
            "inference_seconds": 0.0,
            "total_seconds": time.time() - worker_start,
            "throughput_samples_per_second": 0.0,
            "error": "model_load_failed",
            "scheduler_mode": "dynamic_queue",
            "queue_wait_seconds": 0.0,
            "state": "failed",
        }

    detector = ObjectDetector(loader, prompt_builder=prompt_builder, coord_format=coord_format)
    results: List[Dict[str, Any]] = []
    processed = 0
    failed = 0
    claimed_samples = 0
    completed = 0
    total_queue_wait_seconds = 0.0
    infer_start = time.time()
    progress_bar = None
    last_logged_completed = -1

    if TQDM_AVAILABLE and all_records and live_tqdm_enabled(rank):
        progress_bar = create_inference_progress_bar(
            total=len(all_records),
            model_type=model_type,
            rank=rank,
            physical_gpu=physical_gpu,
            dynamic_mode=True,
        )

    try:
        while True:
            idle_metrics = collect_local_gpu_metrics(local_rank, physical_gpu)
            task_queue.update_worker_status(
                worker_rank=rank,
                gpu_id=physical_gpu,
                state="idle",
                claimed_samples=claimed_samples,
                processed=processed,
                failed=failed,
                total_inference_seconds=time.time() - infer_start,
                queue_wait_seconds=total_queue_wait_seconds,
                **idle_metrics,
            )

            claim_start = time.time()
            sample_indices = task_queue.claim_batch(rank, physical_gpu, queue_batch_size)
            total_queue_wait_seconds += time.time() - claim_start
            if not sample_indices:
                break

            claimed_samples += len(sample_indices)
            current_batch = ",".join(str(index) for index in sample_indices)
            busy_metrics = collect_local_gpu_metrics(local_rank, physical_gpu)
            task_queue.update_worker_status(
                worker_rank=rank,
                gpu_id=physical_gpu,
                state="busy",
                claimed_samples=claimed_samples,
                processed=processed,
                failed=failed,
                current_batch=current_batch,
                total_inference_seconds=time.time() - infer_start,
                queue_wait_seconds=total_queue_wait_seconds,
                **busy_metrics,
            )

            batch_wall_start = time.time()
            prepared = []
            failed_indices = set()
            error_messages: Dict[int, str] = {}
            batch_records = [all_records[index] for index in sample_indices]
            for sample_index, record in zip(sample_indices, batch_records):
                try:
                    image_path = DatasetLoader.extract_image_path(record)
                    query = DatasetLoader.extract_query(record)
                    ground_truth = DatasetLoader.parse_ground_truth(record)
                    image = DatasetLoader.load_image(image_path)
                    if image is None:
                        failed += 1
                        failed_indices.add(sample_index)
                        error_messages[sample_index] = "image_load_failed"
                        continue
                    prepared.append((sample_index, record, image_path, query, ground_truth, image))
                except Exception as exc:
                    failed += 1
                    failed_indices.add(sample_index)
                    error_messages[sample_index] = str(exc)
                    log(f"prepare sample failed on gpu {physical_gpu}: {exc}", rank=rank)

            if prepared:
                images = [item[5] for item in prepared]
                queries = [item[3] for item in prepared]
                det_results = detector.detect_batch(
                    images,
                    queries,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                )
                batch_elapsed = time.time() - batch_wall_start
                per_sample_elapsed = batch_elapsed / len(prepared) if prepared else 0.0

                for (sample_index, record, image_path, query, ground_truth, _image), det_result in zip(prepared, det_results):
                    detections = det_result.get("detections", []) if det_result.get("success") else []
                    if not det_result.get("success", False):
                        failed += 1
                        failed_indices.add(sample_index)
                        error_messages[sample_index] = det_result.get("error", "")
                    metrics = MetricsCalculator.compute_sample_metrics(detections, ground_truth, args.iou_threshold)
                    results.append(
                        {
                            "gpu_id": physical_gpu,
                            "model_type": model_type,
                            "index": record.get("index", 0),
                            "image_path": image_path,
                            "image_width": _image.size[0],
                            "image_height": _image.size[1],
                            "query": query,
                            "detections": detections,
                            "metrics": metrics,
                            "raw_response": det_result.get("raw_response", "") if det_result.get("success") else "",
                            "inference_success": bool(det_result.get("success", False)),
                            "inference_error": det_result.get("error", "") if not det_result.get("success") else "",
                            "sample_wall_time_seconds": round(per_sample_elapsed, 6),
                        }
                    )
                    processed += 1

            batch_elapsed = time.time() - batch_wall_start
            task_queue.complete_batch(
                sample_indices,
                failed_indices=failed_indices,
                error_messages=error_messages,
            )

            completed += len(sample_indices)
            status_metrics = collect_local_gpu_metrics(local_rank, physical_gpu)
            task_queue.update_worker_status(
                worker_rank=rank,
                gpu_id=physical_gpu,
                state="idle",
                claimed_samples=claimed_samples,
                processed=processed,
                failed=failed,
                current_batch="",
                last_batch_seconds=batch_elapsed,
                total_inference_seconds=time.time() - infer_start,
                queue_wait_seconds=total_queue_wait_seconds,
                **status_metrics,
            )

            if progress_bar is not None:
                if rank == 0:
                    snapshot = task_queue.get_progress_snapshot()
                    sync_progress_bar(
                        progress_bar,
                        snapshot["completed"],
                        processed=snapshot["processed"],
                        failed=snapshot["failed"],
                    )
                else:
                    sync_progress_bar(progress_bar, completed, processed=processed, failed=failed)
            if rank == 0:
                snapshot = task_queue.get_progress_snapshot()
                if snapshot["completed"] == last_logged_completed:
                    continue
                last_logged_completed = snapshot["completed"]
                log(
                    format_progress_summary(
                        model_type,
                        total=snapshot["total"],
                        completed=snapshot["completed"],
                        processed=snapshot["processed"],
                        failed=snapshot["failed"],
                        remaining=snapshot["remaining"],
                    ),
                    rank=rank,
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    inference_seconds = time.time() - infer_start
    gpu_metrics = collect_local_gpu_metrics(local_rank, physical_gpu)
    loader.unload_model()
    total_seconds = time.time() - worker_start
    task_queue.update_worker_status(
        worker_rank=rank,
        gpu_id=physical_gpu,
        state="done",
        claimed_samples=claimed_samples,
        processed=processed,
        failed=failed,
        current_batch="",
        last_batch_seconds=0.0,
        total_inference_seconds=inference_seconds,
        queue_wait_seconds=total_queue_wait_seconds,
        **gpu_metrics,
    )
    stats = {
        "rank": rank,
        "gpu_id": physical_gpu,
        "claimed_samples": claimed_samples,
        "processed": processed,
        "failed": failed,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": total_seconds,
        "throughput_samples_per_second": processed / inference_seconds if inference_seconds > 0 else 0.0,
        "error": "",
        "scheduler_mode": "dynamic_queue",
        "queue_wait_seconds": total_queue_wait_seconds,
        "state": "done",
        **gpu_metrics,
    }
    return results, stats


def merge_results(result_dir: Path, model_type: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    partial_dir = result_dir / "partials"
    merged_results: List[Dict[str, Any]] = []
    gpu_stats: Dict[str, Any] = {}
    for partial_path in sorted(partial_dir.glob(f"{model_type}_rank*.json")):
        with open(partial_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        merged_results.extend(payload.get("results", []))
        rank = payload.get("rank", 0)
        gpu_id = payload.get("gpu_id", rank)
        gpu_stats[f"{model_type}_rank{rank}_gpu{gpu_id}"] = payload.get("stats", {})
    merged_results.sort(key=lambda item: item.get("index", 0))
    return merged_results, gpu_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed multi-GPU inference for multimodal model comparison")
    parser.add_argument("--gpu_ids", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_adapter_path", type=str, default="")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--export_labelme", action="store_true")
    parser.add_argument("--scheduler_mode", choices=("static_partition", "dynamic_queue"), default="static_partition")
    parser.add_argument("--partition_strategy", choices=("contiguous", "round_robin"), default="round_robin")
    parser.add_argument("--queue_batch_size", type=int, default=None)
    parser.add_argument("--coord_format", choices=("xyxy", "yxyx", "xywh", "cxcywh"), default="xyxy", help="输出bbox格式: xyxy, yxyx(Gemma4 box_2d), xywh, 或 cxcywh")
    parser.add_argument(
        "--attn_implementation", type=str, default=None, choices=["sdpa", "flash_attention_2", "eager"], help="注意力实现方式: sdpa(推荐), flash_attention_2, eager. None则由Unsloth自动选择"
    )
    parser.add_argument("--enable_compile", action="store_true", default=False, help="启用torch.compile (默认禁用, 仅在推理场景且确认稳定时启用)")
    return parser.parse_args()


def main():
    global INFERENCE_TEMPERATURE, INFERENCE_TOP_P, INFERENCE_MAX_NEW_TOKENS

    args = parse_args()
    args.gpu_ids = [int(part.strip()) for part in args.gpu_ids.split(",") if part.strip()]
    INFERENCE_TEMPERATURE = args.temperature
    INFERENCE_TOP_P = args.top_p
    INFERENCE_MAX_NEW_TOKENS = args.max_new_tokens

    # Resolve prompt builder (matches training prompt format) and coord format
    prompt_builder = build_cn_detection_prompt
    coord_format = args.coord_format

    rank = None
    try:
        rank, local_rank, world_size, is_distributed, physical_gpu = setup_distributed(args.gpu_ids)
        result_dir = Path(args.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        all_records = ensure_record_indices(DatasetLoader.load_jsonl(args.data_path))
        if args.max_eval_samples:
            all_records = all_records[: args.max_eval_samples]

        dynamic_task_payloads = build_dynamic_task_payloads(all_records) if args.scheduler_mode == "dynamic_queue" else []
        partition_records: List[Dict[str, Any]] = []
        if args.scheduler_mode == "static_partition":
            partition_records = get_partition(
                all_records,
                rank,
                world_size,
                strategy=args.partition_strategy,
            )
            log(
                f"worker ready: local_rank={local_rank}, physical_gpu={physical_gpu}, "
                f"world_size={world_size}, assigned_samples={len(partition_records)}, "
                f"scheduler_mode={args.scheduler_mode}, partition_strategy={args.partition_strategy}",
                rank=rank,
            )
        else:
            log(
                f"worker ready: local_rank={local_rank}, physical_gpu={physical_gpu}, "
                f"world_size={world_size}, total_samples={len(all_records)}, "
                f"scheduler_mode={args.scheduler_mode}, queue_batch_size={args.queue_batch_size or args.batch_size}",
                rank=rank,
            )

        total_start = time.time()

        log("round 1 start: finetuned", rank=rank)
        if args.scheduler_mode == "dynamic_queue":
            if rank == 0:
                SQLiteTaskQueue(get_queue_db_path(result_dir, "finetuned")).initialize(
                    dynamic_task_payloads,
                    metadata={
                        "model_type": "finetuned",
                        "scheduler_mode": args.scheduler_mode,
                        "partition_strategy": args.partition_strategy,
                        "world_size": world_size,
                    },
                    reset=True,
                )
            barrier()
            ft_results, ft_stats = run_model_round_dynamic(
                "finetuned",
                all_records,
                args,
                rank,
                local_rank,
                physical_gpu,
                result_dir,
                prompt_builder=prompt_builder,
                coord_format=coord_format,
            )
        else:
            ft_results, ft_stats = run_model_round("finetuned", partition_records, args, rank, local_rank, physical_gpu, prompt_builder=prompt_builder, coord_format=coord_format)
        barrier()
        if args.scheduler_mode == "dynamic_queue" and rank == 0:
            ft_snapshot = validate_dynamic_queue_round(
                result_dir,
                "finetuned",
                expected_total=len(all_records),
                world_size=world_size,
            )
            log(
                f"round 1 queue verified: total={ft_snapshot['total']}, completed={ft_snapshot['completed']}, " f"failed={ft_snapshot['failed']}, remaining={ft_snapshot['remaining']}",
                rank=rank,
            )
        barrier()
        ft_partial = save_partial_results(
            result_dir,
            "finetuned",
            rank,
            {"rank": rank, "gpu_id": physical_gpu, "stats": ft_stats, "results": ft_results},
        )
        log(f"round 1 saved: {ft_partial}", rank=rank)

        barrier()
        if rank == 0:
            ft_partial_summary = wait_for_partial_results(
                result_dir,
                "finetuned",
                expected_parts=world_size,
                expected_total=len(all_records),
            )
            log(
                f"round 1 partials verified: files={ft_partial_summary['partials']}, samples={ft_partial_summary['results']}",
                rank=rank,
            )
        barrier()

        log("round 2 start: base", rank=rank)
        if args.scheduler_mode == "dynamic_queue":
            if rank == 0:
                SQLiteTaskQueue(get_queue_db_path(result_dir, "base")).initialize(
                    dynamic_task_payloads,
                    metadata={
                        "model_type": "base",
                        "scheduler_mode": args.scheduler_mode,
                        "partition_strategy": args.partition_strategy,
                        "world_size": world_size,
                    },
                    reset=True,
                )
            barrier()
            base_results, base_stats = run_model_round_dynamic(
                "base",
                all_records,
                args,
                rank,
                local_rank,
                physical_gpu,
                result_dir,
                prompt_builder=prompt_builder,
                coord_format=coord_format,
            )
        else:
            base_results, base_stats = run_model_round("base", partition_records, args, rank, local_rank, physical_gpu, prompt_builder=prompt_builder, coord_format=coord_format)
        barrier()
        if args.scheduler_mode == "dynamic_queue" and rank == 0:
            base_snapshot = validate_dynamic_queue_round(
                result_dir,
                "base",
                expected_total=len(all_records),
                world_size=world_size,
            )
            log(
                f"round 2 queue verified: total={base_snapshot['total']}, completed={base_snapshot['completed']}, " f"failed={base_snapshot['failed']}, remaining={base_snapshot['remaining']}",
                rank=rank,
            )
        barrier()
        base_partial = save_partial_results(
            result_dir,
            "base",
            rank,
            {"rank": rank, "gpu_id": physical_gpu, "stats": base_stats, "results": base_results},
        )
        log(f"round 2 saved: {base_partial}", rank=rank)

        barrier()
        if rank == 0:
            base_partial_summary = wait_for_partial_results(
                result_dir,
                "base",
                expected_parts=world_size,
                expected_total=len(all_records),
            )
            log(
                f"round 2 partials verified: files={base_partial_summary['partials']}, samples={base_partial_summary['results']}",
                rank=rank,
            )
        barrier()

        if rank == 0:
            merged_ft_results, ft_gpu_stats = merge_results(result_dir, "finetuned")
            merged_base_results, base_gpu_stats = merge_results(result_dir, "base")
            ft_round_total = max((float(stats.get("total_seconds", 0.0)) for stats in ft_gpu_stats.values()), default=0.0)
            base_round_total = max((float(stats.get("total_seconds", 0.0)) for stats in base_gpu_stats.values()), default=0.0)

            ft_result_file = result_dir / "finetuned_results.json"
            base_result_file = result_dir / "base_results.json"
            with open(ft_result_file, "w", encoding="utf-8") as f:
                json.dump(merged_ft_results, f, ensure_ascii=False, indent=2)
            with open(base_result_file, "w", encoding="utf-8") as f:
                json.dump(merged_base_results, f, ensure_ascii=False, indent=2)

            ft_metrics = MetricsCalculator.aggregate_metrics([item["metrics"] for item in merged_ft_results])
            base_metrics = MetricsCalculator.aggregate_metrics([item["metrics"] for item in merged_base_results])
            if args.scheduler_mode == "dynamic_queue":
                ft_queue_report = SQLiteTaskQueue(get_queue_db_path(result_dir, "finetuned")).build_report(
                    total_seconds=ft_round_total,
                    scheduler_mode=args.scheduler_mode,
                    world_size=world_size,
                )
                base_queue_report = SQLiteTaskQueue(get_queue_db_path(result_dir, "base")).build_report(
                    total_seconds=base_round_total,
                    scheduler_mode=args.scheduler_mode,
                    world_size=world_size,
                )
                ft_worker_rows = ft_queue_report["worker_status"]
                base_worker_rows = base_queue_report["worker_status"]
                ft_task_counts = ft_queue_report["task_counts"]
                base_task_counts = base_queue_report["task_counts"]
                ft_metadata = ft_queue_report.get("metadata", {})
                base_metadata = base_queue_report.get("metadata", {})
            else:
                ft_worker_rows = build_worker_rows_from_stats(ft_gpu_stats)
                base_worker_rows = build_worker_rows_from_stats(base_gpu_stats)
                ft_task_counts = {"pending": 0, "claimed": 0, "done": len(merged_ft_results)}
                base_task_counts = {"pending": 0, "claimed": 0, "done": len(merged_base_results)}
                ft_metadata = {
                    "model_type": "finetuned",
                    "scheduler_mode": args.scheduler_mode,
                    "partition_strategy": args.partition_strategy,
                    "world_size": world_size,
                }
                base_metadata = {
                    "model_type": "base",
                    "scheduler_mode": args.scheduler_mode,
                    "partition_strategy": args.partition_strategy,
                    "world_size": world_size,
                }

            ft_load_balance = save_load_balance_artifacts(
                result_dir=result_dir,
                model_type="finetuned",
                scheduler_mode=args.scheduler_mode,
                partition_strategy=args.partition_strategy,
                world_size=world_size,
                total_seconds=ft_round_total,
                merged_results=merged_ft_results,
                worker_rows=ft_worker_rows,
                task_counts=ft_task_counts,
                metadata=ft_metadata,
            )
            base_load_balance = save_load_balance_artifacts(
                result_dir=result_dir,
                model_type="base",
                scheduler_mode=args.scheduler_mode,
                partition_strategy=args.partition_strategy,
                world_size=world_size,
                total_seconds=base_round_total,
                merged_results=merged_base_results,
                worker_rows=base_worker_rows,
                task_counts=base_task_counts,
                metadata=base_metadata,
            )
            summary = {
                "finetuned": ft_metrics,
                "base": base_metrics,
                "gpu_stats": {**ft_gpu_stats, **base_gpu_stats},
                "scheduler_mode": args.scheduler_mode,
                "partition_strategy": args.partition_strategy,
                "queue_batch_size": args.queue_batch_size or args.batch_size,
                "load_balance_reports": {
                    "finetuned": ft_load_balance["comparison"],
                    "base": base_load_balance["comparison"],
                },
                "load_balance_report_files": {
                    "finetuned_json": ft_load_balance["json_path"],
                    "finetuned_markdown": ft_load_balance["markdown_path"],
                    "base_json": base_load_balance["json_path"],
                    "base_markdown": base_load_balance["markdown_path"],
                },
                "total_seconds": time.time() - total_start,
            }

            summary_file = result_dir / "comparison_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            ft_labelme_count = 0
            base_labelme_count = 0
            if args.export_labelme:
                log("starting batch LabelMe export: finetuned", rank=rank)
                ft_export = save_labelme_results(
                    merged_ft_results,
                    result_dir / "labelme_finetuned",
                    image_loader=DatasetLoader.load_image,
                    progress_factory=create_progress_bar if TQDM_AVAILABLE else None,
                    progress_desc="labelme finetuned",
                    log_fn=lambda message: log(message, rank=rank),
                )
                ft_labelme_count = ft_export["written"]

                log("starting batch LabelMe export: base", rank=rank)
                base_export = save_labelme_results(
                    merged_base_results,
                    result_dir / "labelme_base",
                    image_loader=DatasetLoader.load_image,
                    progress_factory=create_progress_bar if TQDM_AVAILABLE else None,
                    progress_desc="labelme base",
                    log_fn=lambda message: log(message, rank=rank),
                )
                base_labelme_count = base_export["written"]

                if ft_export["failed"] or base_export["failed"]:
                    log(
                        "labelme export completed with errors: " f"finetuned_failed={ft_export['failed']}, base_failed={base_export['failed']}",
                        rank=rank,
                    )
            else:
                log("labelme export skipped (--export_labelme not set)", rank=rank)

            log(f"all done in {summary['total_seconds']:.2f}s", rank=rank)
            log(f"finetuned samples={len(merged_ft_results)}, base samples={len(merged_base_results)}", rank=rank)
            log(
                "load balance reports: " f"finetuned={ft_load_balance['markdown_path']}, " f"base={base_load_balance['markdown_path']}",
                rank=rank,
            )
            log(
                "finetuned metrics: " f"precision={ft_metrics.get('mean_precision', 0):.3f}, " f"recall={ft_metrics.get('mean_recall', 0):.3f}, " f"f1={ft_metrics.get('mean_f1', 0):.3f}",
                rank=rank,
            )
            log(
                "base metrics: " f"precision={base_metrics.get('mean_precision', 0):.3f}, " f"recall={base_metrics.get('mean_recall', 0):.3f}, " f"f1={base_metrics.get('mean_f1', 0):.3f}",
                rank=rank,
            )
            log(f"labelme files: finetuned={ft_labelme_count}, base={base_labelme_count}", rank=rank)

        barrier()
    except Exception as exc:
        log_exception("distributed inference failed", exc, rank=rank)
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
