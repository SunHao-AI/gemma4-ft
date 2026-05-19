"""Compatibility wrapper for gemma4_ft.data.labelme.labelme_stats_processor."""
from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("gemma4_ft.data.labelme.labelme_stats_processor")
globals().update(_module.__dict__)
_sys.modules[__name__] = _module
