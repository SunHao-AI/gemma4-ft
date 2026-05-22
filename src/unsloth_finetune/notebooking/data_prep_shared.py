from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from unsloth_finetune.data.labelme import create_progress_bar
from unsloth_finetune.data.labelme.file_utils import json_loads, parse_json_file


def prescan_label_statistics(source_dir: Path, max_workers: int = 4) -> Dict[str, Any]:
    total_files = 0
    skipped_files = 0
    image_url_files = 0
    local_image_files = 0
    json_files = list(source_dir.rglob("*.json"))

    def scan_file(json_file: Path) -> Dict[str, Any]:
        empty_result = {
            "scanned_files": 0,
            "skipped_files": 0,
            "image_url_files": 0,
            "local_image_files": 0,
            "labels": [],
        }
        try:
            data = parse_json_file(json_file)
            if data is None:
                return {**empty_result, "skipped_files": 1}

            shapes = data.get("shapes", [])
            labels = [shape.get("label", "unknown") for shape in shapes if isinstance(shape, dict)]

            return {
                "scanned_files": 1,
                "skipped_files": 0,
                "image_url_files": 1 if data.get("imageUrl") else 0,
                "local_image_files": 1 if (not data.get("imageUrl") and data.get("imagePath")) else 0,
                "labels": labels,
            }
        except Exception:
            return {**empty_result, "skipped_files": 1}

    all_labels: list[str] = []
    progress_bar = create_progress_bar(total=len(json_files), desc="Label预扫描", unit="文件")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(scan_file, json_files):
            total_files += result["scanned_files"]
            skipped_files += result["skipped_files"]
            image_url_files += result["image_url_files"]
            local_image_files += result["local_image_files"]
            all_labels.extend(result["labels"])
            if progress_bar:
                progress_bar.update(1)
    if progress_bar:
        progress_bar.close()

    label_counter = Counter(all_labels)
    return {
        "total_json_files": len(json_files),
        "scanned_files": total_files,
        "skipped_files": skipped_files,
        "image_url_files": image_url_files,
        "local_image_files": local_image_files,
        "label_counts": dict(label_counter.most_common()),
        "total_labels": len(label_counter),
        "total_instances": sum(label_counter.values()),
    }


def verify_unsloth_format(jsonl_path: str | Path, num_samples: int = 3) -> tuple[bool, Dict[str, Any] | str]:
    """验证JSONL文件格式完整性，支持 OpenAI messages 和 ShareGPT 两种格式。"""
    path = Path(jsonl_path)
    if not path.exists():
        return False, "文件不存在"

    valid = 0
    total = 0
    errors: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= num_samples:
                break
            total += 1
            try:
                record = json_loads(line)

                # Detect schema type
                if "conversations" in record:
                    # ShareGPT format
                    conversations = record.get("conversations", [])
                    if not conversations:
                        errors.append(f"样本{index + 1}: conversations为空")
                        continue
                    if conversations[0].get("from") != "human":
                        errors.append(f"样本{index + 1}: conversations第一条应来自human")
                        continue
                    if len(conversations) == 2 and conversations[1].get("from") != "gpt":
                        errors.append(f"样本{index + 1}: conversations第二条应来自gpt")
                        continue
                    if "id" not in record:
                        errors.append(f"样本{index + 1}: 缺少id字段")
                        continue
                    if "image" not in record:
                        errors.append(f"样本{index + 1}: 缺少image字段")
                        continue
                    valid += 1

                elif "messages" in record:
                    # OpenAI messages format
                    messages = record.get("messages", [])
                    if "messages" not in record:
                        errors.append(f"样本{index + 1}: 缺少messages字段")
                        continue
                    if "images" not in record:
                        errors.append(f"样本{index + 1}: 缺少images字段")
                        continue
                    if not messages:
                        errors.append(f"样本{index + 1}: messages为空")
                        continue
                    # Check user message exists
                    if messages[0].get("role") != "user":
                        errors.append(f"样本{index + 1}: messages第一条应为user")
                        continue
                    # Assistant message optional for test-only records
                    if len(messages) > 1 and messages[1].get("role") != "assistant":
                        errors.append(f"样本{index + 1}: messages第二条应为assistant")
                        continue
                    valid += 1

                else:
                    errors.append(f"样本{index + 1}: 未知格式(缺少messages或conversations)")
                    continue

            except ValueError as exc:
                errors.append(f"样本{index + 1}: JSON解析错误 - {exc}")

    passed = valid == total
    result = {
        "检查样本": total,
        "有效样本": valid,
        "结果": "✅ 通过" if passed else "❌ 失败",
    }
    if errors:
        result["错误详情"] = "; ".join(errors[:3])
    return passed, result