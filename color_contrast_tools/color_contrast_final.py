"""Compatibility wrapper for unsloth_finetune.tools.color_contrast.color_contrast_final."""
from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("unsloth_finetune.tools.color_contrast.color_contrast_final")
globals().update(_module.__dict__)

if __name__ == "__main__":
    _module.main()
else:
    _sys.modules[__name__] = _module

