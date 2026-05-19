"""Compatibility wrapper for unsloth_finetune.core.labelme_export."""
from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("unsloth_finetune.core.labelme_export")
globals().update(_module.__dict__)
_sys.modules[__name__] = _module

