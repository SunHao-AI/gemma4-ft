"""Local development shim for the src-based `unsloth_finetune` package."""

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_SRC_PKG_DIR = _PKG_DIR.parent / "src" / "unsloth_finetune"

__path__ = [str(_PKG_DIR)]
if _SRC_PKG_DIR.exists():
    __path__.append(str(_SRC_PKG_DIR))

