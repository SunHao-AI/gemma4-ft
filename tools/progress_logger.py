"""
共享的进度条与日志基础设施模块
提供统一的tqdm进度条和日志系统配置，实现控制台输出与日志文件分离

核心设计原则：
- use_tqdm=True时：控制台仅展示tqdm进度条，所有运行信息写入日志文件
- use_tqdm=False时：控制台展示完整日志信息
- logger.propagate=False：防止日志消息泄漏到根日志记录器
- NullHandler兜底：use_tqdm=True且无log_file时，确保日志不泄漏到stderr

分段进度条功能：
- PhaseProgressManager：管理多阶段任务的进度条
- 每个阶段独立显示进度条，清晰展示任务执行状态
- 支持阶段描述动态切换，实时反馈当前处理阶段
"""

import sys
import logging
from typing import Optional, Set, List, Dict, Any
from datetime import datetime

IN_NOTEBOOK = False
TQDM_AVAILABLE = False
tqdm = None
notebook_tqdm = None

try:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None and "IPKernelApp" in ip.config:
            IN_NOTEBOOK = True
    except Exception:
        pass

    if IN_NOTEBOOK:
        try:
            from tqdm.notebook import tqdm as notebook_tqdm

            TQDM_AVAILABLE = True
        except ImportError:
            pass
    else:
        try:
            from tqdm import tqdm

            TQDM_AVAILABLE = True
        except ImportError:
            pass

except ImportError:
    pass

SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_progress_logging(
    logger_name: str,
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    use_tqdm: bool = True,
) -> logging.Logger:
    """
    配置日志系统，实现控制台输出与日志文件分离

    use_tqdm=True: 控制台仅展示tqdm进度条，所有运行信息写入日志文件
    use_tqdm=False: 控制台展示完整日志信息

    Args:
        logger_name: 日志记录器名称
        log_file: 日志文件路径，None则不记录到文件
        log_level: 日志级别
        use_tqdm: 是否使用tqdm进度条模式

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    if not use_tqdm:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    return logger


def create_progress_bar(
    total: int,
    desc: str = "处理",
    unit: str = "文件",
    mininterval: float = 0.5,
    maxinterval: float = 5.0,
    **kwargs,
):
    """
    创建标准化的tqdm进度条，自动适配Jupyter Notebook环境

    Args:
        total: 总数量
        desc: 进度条描述文字
        unit: 单位描述
        mininterval: 最小更新间隔(秒)，notebook环境默认0.5秒以确保及时更新
        maxinterval: 最大更新间隔(秒)
        **kwargs: 其他tqdm参数

    Returns:
        tqdm进度条实例，tqdm不可用时返回None
    """
    if not TQDM_AVAILABLE:
        return None

    if IN_NOTEBOOK and notebook_tqdm is not None:
        return notebook_tqdm(
            total=total,
            desc=desc,
            unit=unit,
            leave=True,
            mininterval=mininterval,
            maxinterval=maxinterval,
            **kwargs,
        )
    elif tqdm is not None:
        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            position=0,
            leave=True,
            mininterval=mininterval,
            maxinterval=maxinterval,
            file=sys.stderr,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            **kwargs,
        )

    return None


class PhaseProgressManager:
    """
    分段进度条管理器
    
    管理多阶段任务的进度条，每个阶段独立显示进度条：
    - 阶段开始时显示阶段描述和预期进度
    - 阶段执行时实时更新进度
    - 阶段结束时显示完成信息并关闭进度条
    
    Attributes:
        phases: 阶段名称列表
        current_phase_index: 当前阶段索引
        phase_info: 各阶段的详细信息
        start_time: 任务开始时间
    """

    PHASE_DISPLAY_NAMES = {
        "validation": "数据验证",
        "deduplication": "去重处理",
        "copy": "文件复制",
        "format_conversion": "格式转换",
        "integrity_check": "完整性验证",
        "statistics": "统计分析",
        "report_generation": "报告生成",
        "final_summary": "最终汇总",
    }

    def __init__(self, phases: List[str], use_tqdm: bool = True):
        """
        初始化分段进度条管理器
        
        Args:
            phases: 阶段名称列表，如 ["validation", "deduplication", "copy"]
            use_tqdm: 是否使用tqdm进度条
        """
        self.phases = phases
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self.current_phase_index = -1
        self._pbar = None
        self.phase_info: Dict[str, Dict[str, Any]] = {}
        self.start_time = datetime.now()
        self._phase_start_time: Optional[datetime] = None

        for phase in phases:
            self.phase_info[phase] = {
                "display_name": self.PHASE_DISPLAY_NAMES.get(phase, phase),
                "total": 0,
                "completed": 0,
                "start_time": None,
                "end_time": None,
            }

    def _get_display_name(self, phase: str) -> str:
        """获取阶段的显示名称"""
        return self.PHASE_DISPLAY_NAMES.get(phase, phase)

    def start_phase(self, phase: str, total: int) -> None:
        """
        开始一个新阶段
        
        Args:
            phase: 阶段名称
            total: 该阶段的总量
        """
        self._complete_current_phase()

        phase_index = self.phases.index(phase) if phase in self.phases else -1
        self.current_phase_index = phase_index

        self.phase_info[phase]["total"] = total
        self.phase_info[phase]["start_time"] = datetime.now()
        self.phase_info[phase]["completed"] = 0
        self._phase_start_time = datetime.now()

        display_name = self._get_display_name(phase)
        phase_prefix = f"[{phase_index + 1}/{len(self.phases)}]"

        if self.use_tqdm:
            self._pbar = create_progress_bar(
                total=total,
                desc=f"{phase_prefix} {display_name}",
                unit="项",
                mininterval=0.3,
            )
        else:
            print(f"\n{phase_prefix} 开始: {display_name} (共 {total} 项)")

    def update(self, n: int = 1, message: Optional[str] = None) -> None:
        """
        更新当前阶段的进度
        
        Args:
            n: 更新的数量，默认为1
            message: 可选的进度消息
        """
        if self.current_phase_index < 0:
            return

        current_phase = self.phases[self.current_phase_index]
        self.phase_info[current_phase]["completed"] += n

        if self._pbar:
            self._pbar.update(n)
            if message:
                self._pbar.set_postfix_str(message)
        else:
            completed = self.phase_info[current_phase]["completed"]
            total = self.phase_info[current_phase]["total"]
            if total > 0:
                percent = (completed / total) * 100
                print(f"  进度: {completed}/{total} ({percent:.1f}%)")

    def set_description(self, desc: str) -> None:
        """
        设置进度条描述
        
        Args:
            desc: 新的描述文字
        """
        if self._pbar:
            self._pbar.set_description(desc)

    def _complete_current_phase(self) -> None:
        """完成当前阶段（内部方法）"""
        if self.current_phase_index < 0:
            return

        current_phase = self.phases[self.current_phase_index]
        self.phase_info[current_phase]["end_time"] = datetime.now()

        if self._pbar:
            self._pbar.close()
            self._pbar = None

        display_name = self._get_display_name(current_phase)
        completed = self.phase_info[current_phase]["completed"]
        total = self.phase_info[current_phase]["total"]

        if not self.use_tqdm:
            phase_prefix = f"[{self.current_phase_index + 1}/{len(self.phases)}]"
            print(f"{phase_prefix} 完成: {display_name} ({completed}/{total}) ✓")

    def complete_phase(self, phase: Optional[str] = None) -> None:
        """
        完成指定阶段或当前阶段
        
        Args:
            phase: 要完成的阶段名称，None则完成当前阶段
        """
        if phase and phase in self.phases:
            target_index = self.phases.index(phase)
            if target_index == self.current_phase_index:
                self._complete_current_phase()
        elif self.current_phase_index >= 0:
            self._complete_current_phase()

        self.current_phase_index = -1

    def complete_all(self, show_summary: bool = True) -> Dict[str, Any]:
        """
        完成所有阶段并显示汇总
        
        Args:
            show_summary: 是否显示完成汇总
            
        Returns:
            Dict: 包含各阶段执行信息的汇总字典
        """
        self._complete_current_phase()

        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()

        summary = {
            "total_duration": total_duration,
            "phases_completed": len(self.phases),
            "phase_details": {},
        }

        for phase, info in self.phase_info.items():
            if info["start_time"] and info["end_time"]:
                phase_duration = (info["end_time"] - info["start_time"]).total_seconds()
            else:
                phase_duration = 0

            summary["phase_details"][phase] = {
                "display_name": info["display_name"],
                "total": info["total"],
                "completed": info["completed"],
                "duration": phase_duration,
            }

        if show_summary:
            print("\n" + "=" * 60)
            print("全部任务完成！")
            print("=" * 60)
            print(f"总耗时: {total_duration:.2f} 秒")
            print(f"完成阶段数: {len(self.phases)}")
            print("-" * 60)
            print("各阶段耗时:")
            for phase, details in summary["phase_details"].items():
                print(f"  {details['display_name']}: {details['duration']:.2f} 秒 ({details['completed']}/{details['total']})")
            print("=" * 60)

        return summary

    def get_progress_info(self) -> Dict[str, Any]:
        """
        获取当前进度信息
        
        Returns:
            Dict: 包含当前进度信息的字典
        """
        if self.current_phase_index < 0:
            return {"phase": None, "progress": 0, "total": 0}

        current_phase = self.phases[self.current_phase_index]
        info = self.phase_info[current_phase]

        return {
            "phase": current_phase,
            "display_name": info["display_name"],
            "progress": info["completed"],
            "total": info["total"],
            "percent": (info["completed"] / info["total"] * 100) if info["total"] > 0 else 0,
        }


def print_phase_header(phase_name: str, phase_index: int, total_phases: int) -> None:
    """
    打印阶段头信息（用于非tqdm模式）
    
    Args:
        phase_name: 阶段名称
        phase_index: 阶段索引（从0开始）
        total_phases: 总阶段数
    """
    display_name = PhaseProgressManager.PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
    print(f"\n{'=' * 60}")
    print(f"[阶段 {phase_index + 1}/{total_phases}] {display_name}")
    print("-" * 60)


def print_phase_footer(phase_name: str, completed: int, total: int) -> None:
    """
    打印阶段完成信息（用于非tqdm模式）
    
    Args:
        phase_name: 阶段名称
        completed: 完成数量
        total: 总数量
    """
    display_name = PhaseProgressManager.PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
    percent = (completed / total * 100) if total > 0 else 100
    print(f"\n{display_name} 完成: {completed}/{total} ({percent:.1f}%) ✓")
