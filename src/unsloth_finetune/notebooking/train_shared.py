from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


MODE_DIR_MAP = {
    "single": "single_gpu",
    "ddp": "ddp_8gpu",
    "device_map": "devicemap_4group",
    "fsdp": "fsdp_8gpu",
    "auto": "auto_detect",
    "multi_node": "multi_node",
    "compare": "compare",
}


def resolve_mode_subdir(mode: str) -> str:
    return MODE_DIR_MAP.get(mode, mode.lower())


def build_lora_output_base(project_root: Path, model_name_short: str = "unsloth_mm_lora") -> Path:
    return Path(project_root) / "models" / "finetuned" / model_name_short


def build_lora_output_dir(
    project_root: Path,
    training_mode: str,
    train_timestamp: str,
    model_name_short: str = "unsloth_mm_lora",
) -> Path:
    return build_lora_output_base(project_root, model_name_short) / resolve_mode_subdir(training_mode) / train_timestamp


def get_latest_output(base_dir: str | Path, mode: str) -> Optional[str]:
    mode_dir = Path(base_dir) / resolve_mode_subdir(mode)
    latest_file = mode_dir / "latest.txt"
    if not latest_file.exists():
        return None

    timestamp = latest_file.read_text(encoding="utf-8").strip()
    if not timestamp:
        return None
    return str(mode_dir / timestamp)


def resolve_latest_output_dir(
    base_dir: str | Path,
    mode: str,
    fallback_output_dir: str | Path,
) -> str:
    return get_latest_output(base_dir, mode) or str(fallback_output_dir)


def discover_eval_data_path(train_data_path: str | Path) -> tuple[Optional[str], Optional[str]]:
    data_path = Path(train_data_path).resolve()
    candidate_names = ["test.jsonl", "valid.jsonl", "val.jsonl", data_path.name]
    seen = set()
    for filename in candidate_names:
        if filename in seen:
            continue
        seen.add(filename)
        candidate = data_path.with_name(filename)
        if candidate.exists():
            return str(candidate), candidate.stem
    return None, None


def resolve_eval_gpu_ids(
    training_mode: str,
    gpu_count: int,
    gpu_groups: Optional[Iterable[Iterable[int]]] = None,
) -> list[int]:
    if gpu_count <= 0:
        return []

    if training_mode == "device_map" and gpu_groups:
        flat: list[int] = []
        for group in gpu_groups:
            flat.extend(int(gpu_id) for gpu_id in group)
        return sorted(dict.fromkeys(flat))

    visible = list(range(gpu_count))
    if training_mode in {"ddp", "device_map", "fsdp", "auto", "multi_node"} and gpu_count > 1:
        return visible
    return [visible[0]]
