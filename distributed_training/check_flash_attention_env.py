"""Compatibility wrapper for gemma4_ft.training.distributed.check_flash_attention_env."""
from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("gemma4_ft.training.distributed.check_flash_attention_env")
globals().update(_module.__dict__)

if __name__ == "__main__":
    _module.main()
else:
    _sys.modules[__name__] = _module
