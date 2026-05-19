import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

try:
    from gemma4_core.bootstrap import bootstrap_notebook_context
    from gemma4_core.runtime import configure_unsloth_compile_cache
except ModuleNotFoundError:
    DEFAULT_UNSLOTH_CACHE_DIRNAME = "unsloth_compiled_cache"
    PROJECT_MARKER = "pyproject.toml"

    def _resolve_notebook_dir(cwd: Optional[Path] = None, notebook_file: str = "") -> Path:
        env_notebook_dir = os.environ.get("GEMMA4_NOTEBOOK_DIR", "").strip()
        if env_notebook_dir:
            return Path(env_notebook_dir).expanduser().resolve()
        if notebook_file:
            return Path(notebook_file).expanduser().resolve().parent
        return Path(cwd or Path.cwd()).expanduser().resolve()

    def _coerce_start_paths(start_paths: Optional[Iterable[Path]]) -> list[Path]:
        resolved_paths = []
        for candidate in start_paths or []:
            if candidate is None:
                continue
            path = Path(candidate).expanduser().resolve()
            if path.is_file():
                path = path.parent
            resolved_paths.append(path)
        if not resolved_paths:
            resolved_paths.append(Path.cwd().expanduser().resolve())
        return resolved_paths

    def _resolve_project_root(
        start_paths: Optional[Iterable[Path]] = None,
        marker: str = PROJECT_MARKER,
        fallback: Optional[Path] = None,
    ) -> Path:
        for start_path in _coerce_start_paths(start_paths):
            for candidate in [start_path] + list(start_path.parents):
                if (candidate / marker).exists():
                    return candidate
        if fallback is not None:
            return Path(fallback).expanduser().resolve()
        raise FileNotFoundError(f"Unable to locate project root containing {marker!r}")

    def bootstrap_notebook_context(
        notebook_file: str = "",
        cwd: Optional[Path] = None,
        marker: str = PROJECT_MARKER,
    ) -> dict:
        cwd_path = Path(cwd or Path.cwd()).expanduser().resolve()
        notebook_dir = _resolve_notebook_dir(cwd=cwd_path, notebook_file=notebook_file)
        project_root = _resolve_project_root(
            start_paths=[cwd_path, notebook_dir, PROJECT_ROOT],
            marker=marker,
            fallback=notebook_dir.parent,
        )

        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)

        os.environ["GEMMA4_NOTEBOOK_DIR"] = str(notebook_dir)
        if notebook_file:
            os.environ["GEMMA4_NOTEBOOK_FILE"] = str(Path(notebook_file).expanduser().resolve())

        return {
            "NOTEBOOK_DIR": notebook_dir,
            "PROJECT_ROOT": project_root,
            "NOTEBOOK_FILE": os.environ.get("GEMMA4_NOTEBOOK_FILE", ""),
        }

    def configure_unsloth_compile_cache(
        base_dir: Path,
        cache_dir_name: str = DEFAULT_UNSLOTH_CACHE_DIRNAME,
    ) -> Path:
        cache_dir = Path(base_dir).expanduser().resolve() / cache_dir_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        os.environ["GEMMA4_UNSLOTH_COMPILE_CACHE_DIR"] = str(cache_dir)
        os.environ["UNSLOTH_COMPILE_LOCATION"] = str(cache_dir)

        try:
            import unsloth_zoo.compiler as unsloth_compiler

            unsloth_compiler.UNSLOTH_COMPILE_LOCATION = str(cache_dir)
        except Exception:
            pass

        return cache_dir


def initialize_notebook_context(
    notebook_file: str = "",
    cwd: Optional[Path] = None,
    configure_unsloth_cache: bool = False,
) -> Dict[str, Any]:
    context = bootstrap_notebook_context(notebook_file=notebook_file, cwd=cwd)
    if configure_unsloth_cache:
        context["UNSLOTH_CACHE_DIR"] = configure_unsloth_compile_cache(context["NOTEBOOK_DIR"])
    return context
