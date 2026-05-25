#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中文字体检测、安装、下载与注册工具。

为 matplotlib 图表提供中文字体支持，支持多策略自动获取：
1. 检查 matplotlib fontManager 已注册的中文字体
2. 扫描系统字体目录并注册
3. 通过系统包管理器安装（Linux/macOS）
4. 通过 HTTP 下载 Noto Sans CJK SC 作为最终回退
"""

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.font_manager as fm

_FONT_CACHE_DIR = Path.home() / ".cache" / "unsloth_fonts"

_NOTO_SANS_SC_URLS = [
    "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/"
    "NotoSansCJKsc-Regular.otf",
    "https://cdn.jsdelivr.net/gh/googlefonts/noto-cjk@main/Sans/OTF/SimplifiedChinese/"
    "NotoSansCJKsc-Regular.otf",
]
_NOTO_SANS_SC_FILE = "NotoSansCJKsc-Regular.otf"

_CHINESE_FONT_NAMES = [
    "SimHei",
    "Microsoft YaHei",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Noto Sans CJK SC",
    "Noto Sans CJK",
    "Source Han Sans SC",
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "Arial Unicode MS",
]


def detect_os() -> str:
    system = platform.system().lower()
    if system == "linux":
        try:
            with open("/etc/os-release", "r") as f:
                content = f.read().lower()
                if "ubuntu" in content or "debian" in content:
                    return "linux_debian"
                elif "centos" in content or "rhel" in content or "fedora" in content:
                    return "linux_redhat"
                elif "arch" in content or "manjaro" in content:
                    return "linux_arch"
                elif "alpine" in content:
                    return "linux_alpine"
        except FileNotFoundError:
            pass
        return "linux_generic"
    elif system == "windows":
        return "windows"
    elif system == "darwin":
        return "macos"
    return "unknown"


def detect_package_manager() -> Optional[str]:
    package_managers = {
        "apt": ["apt-get", "apt"],
        "yum": ["yum", "dnf"],
        "pacman": ["pacman"],
        "apk": ["apk"],
        "brew": ["brew"],
    }
    for pm_name, commands in package_managers.items():
        for cmd in commands:
            if shutil.which(cmd):
                return pm_name
    return None


def install_chinese_font() -> Tuple[bool, str]:
    os_type = detect_os()
    pm = detect_package_manager()

    install_commands = {
        "apt": {
            "install": ["apt-get", "install", "-y", "fonts-wqy-microhei", "fonts-wqy-zenhei", "fonts-noto-cjk"],
            "name": "WenQuanYi / Noto CJK",
        },
        "yum": {
            "install": ["yum", "install", "-y", "wqy-microhei-fonts", "wqy-zenhei-fonts", "google-noto-sans-cjk-fonts"],
            "name": "WenQuanYi / Noto CJK",
        },
        "dnf": {
            "install": ["dnf", "install", "-y", "wqy-microhei-fonts", "wqy-zenhei-fonts", "google-noto-sans-cjk-fonts"],
            "name": "WenQuanYi / Noto CJK",
        },
        "pacman": {
            "install": ["pacman", "-S", "--noconfirm", "wqy-microhei", "wqy-zenhei", "noto-fonts-cjk"],
            "name": "WenQuanYi / Noto CJK",
        },
        "apk": {
            "install": ["apk", "add", "font-wqy-microhei"],
            "name": "WenQuanYi Micro Hei",
        },
        "brew": {
            "install": ["brew", "install", "font-wqy-microhei", "font-wqy-zenhei"],
            "name": "WenQuanYi",
        },
    }

    if os_type == "windows":
        return False, "Windows系统通常自带中文字体(如微软雅黑)，无需安装。如缺少请手动下载安装。"

    if os_type == "macos":
        if pm == "brew":
            cmd_info = install_commands["brew"]
            print("检测到 macOS 系统，尝试使用 Homebrew 安装中文字体...")
            try:
                subprocess.run(cmd_info["install"], check=True, capture_output=True)
                subprocess.run(["fc-cache", "-fv"], check=True, capture_output=True)
                return True, f"成功安装 {cmd_info['name']} 字体"
            except subprocess.CalledProcessError as e:
                return False, f"安装失败: {e.stderr.decode() if e.stderr else str(e)}"
            except FileNotFoundError:
                return False, "未找到 Homebrew，请手动安装中文字体或安装 Homebrew"
        return False, "macOS 通常自带中文字体。如缺少请手动安装或使用 Homebrew: brew install font-wqy-microhei"

    if os_type.startswith("linux"):
        if pm and pm in install_commands:
            cmd_info = install_commands[pm]
            print(f"检测到 Linux 系统 ({os_type})，包管理器: {pm}")
            print(f"尝试安装中文字体: {cmd_info['name']}...")

            try:
                if pm == "apt":
                    subprocess.run(["apt-get", "update"], check=True, capture_output=True, timeout=60)
                subprocess.run(cmd_info["install"], check=True, capture_output=True, timeout=120)
                subprocess.run(["fc-cache", "-fv"], check=True, capture_output=True, timeout=30)
                return True, f"成功安装 {cmd_info['name']} 字体，已刷新字体缓存"
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.decode() if e.stderr else str(e)
                if "Permission denied" in error_msg or "not root" in error_msg.lower():
                    hint = (
                        f"需要 root 权限安装字体，请使用 sudo 运行此脚本"
                        f"或手动执行:\n  sudo {cmd_info['install'][0]}"
                        f" install {cmd_info['install'][-1]}"
                    )
                    return False, hint
                return False, f"安装失败: {error_msg}"
            except subprocess.TimeoutExpired:
                return False, "安装超时，请手动安装字体"
            except FileNotFoundError as e:
                return False, f"未找到包管理器命令: {e}"
        else:
            manual_cmds = {
                "linux_debian": "sudo apt-get install fonts-wqy-microhei fonts-noto-cjk",
                "linux_redhat": "sudo yum install wqy-microhei-fonts google-noto-sans-cjk-fonts",
                "linux_arch": "sudo pacman -S wqy-microhei noto-fonts-cjk",
                "linux_alpine": "apk add font-wqy-microhei",
                "linux_generic": "请根据您的 Linux 发行版安装中文字体包",
            }
            manual_hint = (
                f"未检测到支持的包管理器，请手动安装:\n"
                f"  {manual_cmds.get(os_type, manual_cmds['linux_generic'])}"
            )
            return False, manual_hint

    return False, "未知操作系统，请手动安装中文字体"


def get_system_font_dirs() -> list:
    system = platform.system()
    font_dirs = []

    if system == "Linux":
        font_dirs = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            "/usr/share/fonts/truetype",
            "/usr/share/fonts/opentype",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts"),
        ]
    elif system == "Darwin":
        font_dirs = [
            "/System/Library/Fonts",
            "/Library/Fonts",
            "/System/Library/Fonts/Supplemental",
            os.path.expanduser("~/Library/Fonts"),
        ]
    elif system == "Windows":
        font_dirs = [
            os.path.join(
                os.environ.get("SystemRoot", "C:\\Windows"), "Fonts"
            ),
            os.path.join(
                os.environ.get("LOCALAPPDATA", ""),
                "Microsoft",
                "Windows",
                "Fonts",
            ) if os.environ.get("LOCALAPPDATA") else None,
        ]
        font_dirs = [d for d in font_dirs if d]

    return [d for d in font_dirs if os.path.isdir(d)]


def scan_chinese_font_files() -> list:
    chinese_font_patterns = [
        "wqy-microhei", "wqy-zenhei", "WenQuanYi",
        "NotoSansCJK", "NotoSansSC", "SourceHanSans",
        "SimHei", "msyh", "MicrosoftYaHei",
        "PingFang", "Heiti", "STHeiti",
        "Hiragino", "ArialUnicode",
    ]

    font_extensions = [".ttf", ".ttc", ".otf", ".TTF", ".TTC", ".OTF"]
    font_dirs = get_system_font_dirs()
    found_fonts = []

    for font_dir in font_dirs:
        try:
            for root, dirs, files in os.walk(font_dir):
                for file in files:
                    file_lower = file.lower()
                    if any(file_lower.endswith(ext.lower()) for ext in font_extensions):
                        for pattern in chinese_font_patterns:
                            if pattern.lower() in file_lower:
                                full_path = os.path.join(root, file)
                                found_fonts.append(full_path)
                                break
        except Exception:
            continue

    return found_fonts


def register_font_to_matplotlib(font_path: str) -> Optional[str]:
    try:
        prop = fm.FontProperties(fname=font_path)
        font_name = prop.get_name()

        if not any(f.fname == font_path for f in fm.fontManager.ttflist):
            fm.fontManager.addfont(font_path)
            print(f"已注册字体文件: {font_path} (名称: {font_name})")

        return font_name
    except Exception as e:
        print(f"注册字体失败: {font_path}, 错误: {e}")
        return None


def find_and_register_chinese_font() -> Optional[str]:
    font_files = scan_chinese_font_files()

    if font_files:
        print(f"扫描到 {len(font_files)} 个中文字体文件")

        priority_fonts = ["wqy-microhei", "NotoSansCJK", "WenQuanYi"]
        for priority in priority_fonts:
            for font_file in font_files:
                if priority.lower() in font_file.lower():
                    font_name = register_font_to_matplotlib(font_file)
                    if font_name:
                        return font_name

        font_name = register_font_to_matplotlib(font_files[0])
        if font_name:
            return font_name

    return None


def refresh_matplotlib_font_cache():
    cache_dir = None
    try:
        if hasattr(fm, "get_cachedir"):
            cache_dir = fm.get_cachedir()
        elif hasattr(fm, "cachedir"):
            cache_dir = str(fm.cachedir)
        else:
            import matplotlib
            cache_dir = os.path.join(matplotlib.get_data_path(), ".cache")

        if cache_dir and os.path.exists(cache_dir):
            for cache_file in os.listdir(cache_dir):
                if cache_file.startswith("fontlist") and cache_file.endswith(".json"):
                    full_path = os.path.join(cache_dir, cache_file)
                    os.remove(full_path)
                    print(f"已清除 matplotlib 字体缓存: {full_path}")
    except Exception as e:
        print(f"清除字体缓存失败: {e}")

    try:
        if hasattr(fm, "_load_fontmanager"):
            import inspect
            sig = inspect.signature(fm._load_fontmanager)
            if "try_read_cache" in sig.parameters:
                fm._load_fontmanager(try_read_cache=False)
            elif "try_read_cached" in sig.parameters:
                fm._load_fontmanager(try_read_cached=False)
            else:
                fm._load_fontmanager()
        else:
            fm.fontManager = fm.FontManager()
        print("已刷新 matplotlib 字体管理器")
    except Exception as e:
        print(f"刷新字体管理器失败: {e}")
        fm.fontManager = fm.FontManager()


def _validate_font_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 100:
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        return header in (b"OTTO", b"\x00\x01\x00\x00", b"ttcf")
    except Exception:
        return False


def _download_font(url: str, dest: Path, timeout: int = 30) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        import requests
        resp = requests.get(url, timeout=timeout, stream=True)
        if resp.status_code != 200:
            return False
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return _validate_font_file(dest)
    except Exception:
        return False


def download_and_register_chinese_font() -> Optional[str]:
    cached_font = _FONT_CACHE_DIR / _NOTO_SANS_SC_FILE

    if cached_font.exists() and not _validate_font_file(cached_font):
        print(f"缓存字体文件损坏 (大小: {cached_font.stat().st_size} bytes), 正在删除并重新下载")
        cached_font.unlink(missing_ok=True)

    if not _validate_font_file(cached_font):
        print("未找到系统中文字体, 正在尝试下载 Noto Sans CJK SC ...")
        for url in _NOTO_SANS_SC_URLS:
            if _download_font(url, cached_font):
                print(f"字体下载成功: {cached_font}")
                break
            else:
                cached_font.unlink(missing_ok=True)
        if not _validate_font_file(cached_font):
            print("字体下载失败 (所有镜像源均不可达)")
            return None

    if _validate_font_file(cached_font):
        try:
            fm.fontManager.addfont(str(cached_font))
            cached_str = str(cached_font)
            for entry in fm.fontManager.ttflist:
                if cached_str in entry.fname:
                    return entry.name
        except (OSError, RuntimeError, ValueError) as e:
            print(f"字体注册异常: {e}")

    return None


def get_chinese_font(auto_install: bool = True, auto_download: bool = True) -> Optional[str]:
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in _CHINESE_FONT_NAMES:
        if font in available_fonts:
            return font

    font_name = find_and_register_chinese_font()
    if font_name:
        return font_name

    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in _CHINESE_FONT_NAMES:
        if font in available_fonts:
            return font

    if auto_install:
        print("\n" + "=" * 60)
        print("未检测到中文字体，尝试自动安装...")
        print("=" * 60)

        success, message = install_chinese_font()
        print(f"安装结果: {message}")

        if success:
            print("刷新 matplotlib 字体缓存...")
            refresh_matplotlib_font_cache()

            font_name = find_and_register_chinese_font()
            if font_name:
                print(f"成功检测到中文字体: {font_name}")
                return font_name

            available_fonts = [f.name for f in fm.fontManager.ttflist]
            for font in _CHINESE_FONT_NAMES:
                if font in available_fonts:
                    print(f"成功检测到中文字体: {font}")
                    return font

            print("警告: 字体安装成功但仍未检测到，可能需要重启程序")

    if auto_download:
        print("\n" + "=" * 60)
        print("尝试通过 HTTP 下载中文字体...")
        print("=" * 60)

        font_name = download_and_register_chinese_font()
        if font_name:
            print(f"成功下载并注册中文字体: {font_name}")
            return font_name

        available_fonts = [f.name for f in fm.fontManager.ttflist]
        for font in _CHINESE_FONT_NAMES:
            if font in available_fonts:
                return font

    return None


def setup_chinese_font(auto_install: bool = True, auto_download: bool = True) -> Optional[str]:
    """初始化中文字体并配置 matplotlib，返回字体名称。

    应在生成图表前首次调用，替代模块级初始化。
    """
    import matplotlib.pyplot as plt

    font_name = get_chinese_font(auto_install=auto_install, auto_download=auto_download)

    if font_name:
        print(f"使用中文字体: {font_name}")
        plt.rcParams["font.sans-serif"] = [font_name]
        plt.rcParams["axes.unicode_minus"] = False
    else:
        print("警告: 无法使用中文字体，图表将使用英文标签")
        plt.rcParams["font.family"] = "DejaVu Sans"

    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["figure.dpi"] = 100

    return font_name


def get_font_cache_dir() -> Path:
    return _FONT_CACHE_DIR


def get_cached_font_path() -> Optional[Path]:
    cached_font = _FONT_CACHE_DIR / _NOTO_SANS_SC_FILE
    if _validate_font_file(cached_font):
        return cached_font
    return None
