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

    def scan_file(json_file: Path) -> list[str]:
        nonlocal total_files, skipped_files, image_url_files, local_image_files
        try:
            data = parse_json_file(json_file)
            if data is None:
                skipped_files += 1
                return []

            shapes = data.get("shapes", [])
            labels = [shape.get("label", "unknown") for shape in shapes if isinstance(shape, dict)]
            total_files += 1

            if data.get("imageUrl"):
                image_url_files += 1
            elif data.get("imagePath"):
                local_image_files += 1
            return labels
        except Exception:
            skipped_files += 1
            return []

    all_labels: list[str] = []
    progress_bar = create_progress_bar(total=len(json_files), desc="Label预扫描", unit="文件")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for labels in executor.map(scan_file, json_files):
            all_labels.extend(labels)
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
                messages = record.get("messages", [])
                if "messages" not in record:
                    errors.append(f"样本{index + 1}: 缺少messages字段")
                    continue
                if "images" not in record:
                    errors.append(f"样本{index + 1}: 缺少images字段")
                    continue
                if len(messages) != 2 or messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
                    errors.append(f"样本{index + 1}: messages格式不符合要求")
                    continue
                valid += 1
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

