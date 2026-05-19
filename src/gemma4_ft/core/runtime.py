import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover
    ZoneInfo = None
    ZoneInfoNotFoundError = Exception


DEFAULT_LOG_TIMEZONE = "Asia/Shanghai"
DEFAULT_UNSLOTH_CACHE_DIRNAME = "unsloth_compiled_cache"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S %Z%z"


def resolve_notebook_dir(cwd: Optional[Path] = None, notebook_file: str = "") -> Path:
    """Resolve the real notebook directory instead of relying on the process cwd."""
    env_notebook_dir = os.environ.get("GEMMA4_NOTEBOOK_DIR", "").strip()
    if env_notebook_dir:
        return Path(env_notebook_dir).expanduser().resolve()
    if notebook_file:
        return Path(notebook_file).expanduser().resolve().parent
    return Path(cwd or Path.cwd()).expanduser().resolve()


def get_log_timezone_name() -> str:
    return os.environ.get("GEMMA4_LOG_TIMEZONE", DEFAULT_LOG_TIMEZONE).strip() or DEFAULT_LOG_TIMEZONE


def get_log_timezone():
    timezone_name = get_log_timezone_name()
    if ZoneInfo is not None:
        try:
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            pass
    return datetime.now().astimezone().tzinfo or timezone.utc


def get_aware_now() -> datetime:
    return datetime.now(tz=get_log_timezone())


def format_log_timestamp(dt: Optional[datetime] = None) -> str:
    current = dt or get_aware_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=get_log_timezone())
    return current.astimezone(get_log_timezone()).strftime(DEFAULT_LOG_DATE_FORMAT)


def format_file_timestamp(dt: Optional[datetime] = None) -> str:
    current = dt or get_aware_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=get_log_timezone())
    return current.astimezone(get_log_timezone()).strftime("%Y%m%d_%H%M%S%z")


class TimezoneAwareFormatter(logging.Formatter):
    """Logging formatter that renders timestamps in the configured project timezone."""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=get_log_timezone())
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()


def build_logging_formatter(
    fmt: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_LOG_DATE_FORMAT,
) -> TimezoneAwareFormatter:
    return TimezoneAwareFormatter(fmt=fmt, datefmt=datefmt)


def configure_root_logging(
    level: int,
    fmt: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt: str = DEFAULT_LOG_DATE_FORMAT,
) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(build_logging_formatter(fmt=fmt, datefmt=datefmt))
    logging.basicConfig(level=level, handlers=[handler], force=True)


def configure_unsloth_compile_cache(
    base_dir: Path,
    cache_dir_name: str = DEFAULT_UNSLOTH_CACHE_DIRNAME,
) -> Path:
    """Pin Unsloth's compiled cache to an absolute, deterministic directory."""
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
