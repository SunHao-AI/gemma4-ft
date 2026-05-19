from pathlib import Path
from typing import Any, Dict, Optional

from gemma4_ft.core.bootstrap import bootstrap_notebook_context
from gemma4_ft.core.runtime import configure_unsloth_compile_cache


def initialize_notebook_context(
    notebook_file: str = "",
    cwd: Optional[Path] = None,
    configure_unsloth_cache: bool = False,
) -> Dict[str, Any]:
    context = bootstrap_notebook_context(notebook_file=notebook_file, cwd=cwd)
    if configure_unsloth_cache:
        context["UNSLOTH_CACHE_DIR"] = configure_unsloth_compile_cache(context["NOTEBOOK_DIR"])
    return context
