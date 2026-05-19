#!/usr/bin/env python
"""诊断当前环境的 Flash Attention 2 / xFormers / PyTorch CUDA 兼容性。"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def safe_import(module_name: str):
    try:
        module = importlib.import_module(module_name)
        return module, None
    except Exception as exc:  # pragma: no cover - 诊断脚本需要完整异常文本
        return None, str(exc)


def safe_version(package_name: str):
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return None


def run_command(command: list[str]) -> dict:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - 诊断脚本需要完整异常文本
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


def run_flash_attn_smoke_test(torch_module) -> dict:
    result = {"ok": False, "error": None}
    if not torch_module.cuda.is_available():
        result["error"] = "CUDA不可用"
        return result

    try:
        from flash_attn import flash_attn_func

        device = "cuda"
        q = torch_module.randn(2, 128, 8, 64, device=device, dtype=torch_module.float16)
        k = torch_module.randn(2, 128, 8, 64, device=device, dtype=torch_module.float16)
        v = torch_module.randn(2, 128, 8, 64, device=device, dtype=torch_module.float16)
        output = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        result["ok"] = tuple(output.shape) == (2, 128, 8, 64)
    except Exception as exc:  # pragma: no cover - 诊断脚本需要完整异常文本
        result["error"] = str(exc)
    return result


def main():
    report: dict = {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": {
            "torch": safe_version("torch"),
            "flash-attn": safe_version("flash-attn"),
            "xformers": safe_version("xformers"),
            "unsloth": safe_version("unsloth"),
            "triton": safe_version("triton"),
        },
        "nvidia_smi": run_command(["nvidia-smi"]),
    }

    torch_module, torch_error = safe_import("torch")
    report["torch"] = {"import_error": torch_error}
    if torch_module is not None:
        cuda_available = torch_module.cuda.is_available()
        cuda_device_count = torch_module.cuda.device_count() if cuda_available else 0
        report["torch"].update(
            {
                "version": torch_module.__version__,
                "cuda_version": getattr(torch_module.version, "cuda", None),
                "cudnn_version": torch_module.backends.cudnn.version() if hasattr(torch_module.backends, "cudnn") else None,
                "cuda_available": cuda_available,
                "cuda_device_count": cuda_device_count,
                "bf16_supported": torch_module.cuda.is_bf16_supported() if cuda_available else False,
                "flash_sdp_enabled": (
                    torch_module.backends.cuda.flash_sdp_enabled()
                    if cuda_available and hasattr(torch_module.backends.cuda, "flash_sdp_enabled")
                    else None
                ),
                "mem_efficient_sdp_enabled": (
                    torch_module.backends.cuda.mem_efficient_sdp_enabled()
                    if cuda_available and hasattr(torch_module.backends.cuda, "mem_efficient_sdp_enabled")
                    else None
                ),
            }
        )

        devices = []
        for idx in range(cuda_device_count):
            props = torch_module.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
        report["torch"]["devices"] = devices

    flash_attn_module, flash_attn_error = safe_import("flash_attn")
    report["flash_attn"] = {
        "importable": flash_attn_module is not None,
        "import_error": flash_attn_error,
        "module_file": getattr(flash_attn_module, "__file__", None),
    }

    if torch_module is not None and flash_attn_module is not None:
        report["flash_attn"]["smoke_test"] = run_flash_attn_smoke_test(torch_module)

    xformers_module, xformers_error = safe_import("xformers")
    report["xformers"] = {
        "importable": xformers_module is not None,
        "import_error": xformers_error,
        "module_file": getattr(xformers_module, "__file__", None),
    }

    output_path = Path("flash_attention_env_report.json")
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n报告已写入: {output_path.resolve()}")


if __name__ == "__main__":
    main()
