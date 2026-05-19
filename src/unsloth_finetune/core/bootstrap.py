import os
import sys
from pathlib import Path
from typing import Iterable, Optional

from unsloth_finetune.core.runtime import resolve_notebook_dir

PROJECT_MARKER = "pyproject.toml"


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


def resolve_project_root(
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


def ensure_project_root_on_path(
    start_paths: Optional[Iterable[Path]] = None,
    marker: str = PROJECT_MARKER,
    fallback: Optional[Path] = None,
) -> Path:
    project_root = resolve_project_root(start_paths=start_paths, marker=marker, fallback=fallback)
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def bootstrap_notebook_context(
    notebook_file: str = "",
    cwd: Optional[Path] = None,
    marker: str = PROJECT_MARKER,
) -> dict:
    cwd_path = Path(cwd or Path.cwd()).expanduser().resolve()
    notebook_dir = resolve_notebook_dir(cwd=cwd_path, notebook_file=notebook_file)
    start_paths = [cwd_path, notebook_dir]

    project_root = ensure_project_root_on_path(
        start_paths=start_paths,
        marker=marker,
        fallback=notebook_dir.parent,
    )

    os.environ["UNSLOTH_NOTEBOOK_DIR"] = str(notebook_dir)
    os.environ["GEMMA4_NOTEBOOK_DIR"] = str(notebook_dir)
    if notebook_file:
        notebook_file_value = str(Path(notebook_file).expanduser().resolve())
        os.environ["UNSLOTH_NOTEBOOK_FILE"] = notebook_file_value
        os.environ["GEMMA4_NOTEBOOK_FILE"] = notebook_file_value

    return {
        "NOTEBOOK_DIR": notebook_dir,
        "PROJECT_ROOT": project_root,
        "NOTEBOOK_FILE": os.environ.get("UNSLOTH_NOTEBOOK_FILE", "")
        or os.environ.get("GEMMA4_NOTEBOOK_FILE", ""),
    }

