import json
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

LORA_KEY_PATTERN = re.compile(
    r"\.(lora_(?:A|B)|lora_embedding_(?:A|B))(?:\.[^.]+)?\.(?:weight|bias)$"
)


def extract_target_modules_from_state_keys(state_keys: List[str]) -> List[str]:
    target_modules = set()
    for key in state_keys:
        module_name = LORA_KEY_PATTERN.sub("", key)
        if module_name == key:
            continue
        if module_name.startswith("base_model.model."):
            module_name = module_name[len("base_model.model.") :]
        target_modules.add(module_name)
    return sorted(target_modules)


def _find_adapter_weight_file(adapter_dir: Path) -> Optional[Path]:
    for pattern in ("adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt"):
        candidate = adapter_dir / pattern
        if candidate.exists():
            return candidate
    return None


def _load_adapter_state_dict(adapter_dir: Path) -> Dict[str, Any]:
    weight_file = _find_adapter_weight_file(adapter_dir)
    if weight_file is None:
        raise FileNotFoundError(f"adapter 权重文件不存在: {adapter_dir}")

    if weight_file.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(weight_file), device="cpu")

    import torch

    return torch.load(weight_file, map_location="cpu")


def normalize_saved_adapter_config(adapter_dir: str) -> List[str]:
    """Rewrite adapter_config target_modules to the exact modules present in saved weights."""
    adapter_path = Path(adapter_dir).expanduser().resolve()
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return []

    state_dict = _load_adapter_state_dict(adapter_path)
    target_modules = extract_target_modules_from_state_keys(list(state_dict.keys()))
    if not target_modules:
        return []

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("target_modules") == target_modules:
        return target_modules

    config["target_modules"] = target_modules
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return target_modules


@contextmanager
def prepared_adapter_dir(adapter_dir: str) -> Iterator[Path]:
    """Yield an adapter directory whose config matches the actual saved LoRA keys."""
    source_dir = Path(adapter_dir).expanduser().resolve()
    config_path = source_dir / "adapter_config.json"
    if not config_path.exists():
        yield source_dir
        return

    temp_dir = tempfile.TemporaryDirectory(prefix="gemma4_adapter_")
    temp_path = Path(temp_dir.name)
    try:
        for item in source_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, temp_path / item.name)
        normalize_saved_adapter_config(str(temp_path))
        yield temp_path
    finally:
        temp_dir.cleanup()
