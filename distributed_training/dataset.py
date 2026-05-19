"""Compatibility wrapper for unsloth_finetune.training.distributed.dataset."""
from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("unsloth_finetune.training.distributed.dataset")
globals().update(_module.__dict__)
_sys.modules[__name__] = _module

