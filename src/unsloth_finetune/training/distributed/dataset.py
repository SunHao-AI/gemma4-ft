"""多模态数据集模块 - Unsloth 视觉微调

参考 PyTorch Dataset 设计模式，封装数据加载与预处理流程，
与 Unsloth SFTTrainer / HuggingFace Dataset 无缝集成。

数据格式 (Unsloth Vision 标准):
    # 官方格式 - 图片嵌入在 messages content 中:
    {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": PIL.Image.Image},
                {"type": "text", "text": "..."}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "..."}
            ]}
        ]
    }
    # 注意：没有单独的 images 列

优化特性:
    - 进度条显示: 使用 tqdm 显示图片加载和数据转换进度
    - 分批加载: 支持分批处理以控制内存占用
    - 内存监控: 实时显示内存使用情况
    - 延迟加载: 支持 lazy 模式按需加载图片，避免一次性内存暴涨
"""

import gc
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union

from PIL import Image as PILImage
from datasets import Dataset

try:
    from unsloth_finetune.data.labelme.progress_logger import TQDM_AVAILABLE, create_progress_bar
except ImportError:
    try:
        from tqdm import tqdm as _tqdm_class

        TQDM_AVAILABLE = True

        def create_progress_bar(total=None, desc="", unit="it", **kwargs):
            return _tqdm_class(total=total, desc=desc, unit=unit, **kwargs)

    except ImportError:
        TQDM_AVAILABLE = False

        def create_progress_bar(total=None, desc="", unit="it", **kwargs):
            return None


logger = logging.getLogger(__name__)


def get_memory_usage() -> dict:
    """获取当前内存使用情况

    Returns:
        dict: 包含内存使用信息的字典
    """
    try:
        import psutil

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_gb = mem_info.rss / (1024**3)
        vms_gb = mem_info.vms / (1024**3)

        system_mem = psutil.virtual_memory()
        total_gb = system_mem.total / (1024**3)
        available_gb = system_mem.available / (1024**3)
        used_percent = system_mem.percent

        return {
            "process_rss_gb": round(rss_gb, 2),
            "process_vms_gb": round(vms_gb, 2),
            "system_total_gb": round(total_gb, 2),
            "system_available_gb": round(available_gb, 2),
            "system_used_percent": used_percent,
        }
    except ImportError:
        return {
            "process_rss_gb": "N/A (psutil未安装)",
            "process_vms_gb": "N/A",
            "system_total_gb": "N/A",
            "system_available_gb": "N/A",
            "system_used_percent": "N/A",
        }


def print_memory_status(prefix: str = "内存状态") -> None:
    """打印当前内存使用状态

    Args:
        prefix: 打印前缀文字
    """
    mem = get_memory_usage()
    if isinstance(mem["process_rss_gb"], float):
        print(f"{prefix}: 进程RSS={mem['process_rss_gb']}GB, " f"系统可用={mem['system_available_gb']}GB " f"(使用率{mem['system_used_percent']}%)")
    else:
        print(f"{prefix}: {mem['process_rss_gb']}")


class MultimodalDataset:
    """多模态数据集 - Unsloth 视觉微调

    参考 PyTorch torch.utils.data.Dataset 设计模式:
    - __len__: 返回数据集大小
    - __getitem__: 按索引获取单条数据
    - 支持 map-style 和 iterable-style 数据访问

    图片加载策略:
    - preload: 一次性预加载所有图片到内存，适合小中型数据集
    - lazy: 按需加载，适合大型数据集或内存受限场景（推荐）
    - batch: 分批预加载，适合大型数据集且需要预加载的场景

    与 Unsloth SFTTrainer 集成:
    - to_conversation_list(): 返回官方格式的 list of dicts
    - to_hf_dataset(): 返回 HuggingFace Dataset（图片嵌入 messages 中）
    - 输出格式符合 SFTTrainer 的 vision 微调要求
    """

    def __init__(
        self,
        data_path: str,
        image_load_mode: str = "lazy",
        max_workers: int = 8,
        show_progress: bool = True,
        batch_size: Optional[int] = None,
        image_size: Optional[tuple[int, int]] = None,
    ):
        """
        Args:
            data_path: JSONL 数据文件路径
            image_load_mode: 图片加载策略 ("preload", "lazy", "batch")
            max_workers: 预加载模式下的线程池大小
            show_progress: 是否显示进度条
            batch_size: 分批加载时的批次大小 (仅 batch 模式有效)
            image_size: 训练前统一重采样尺寸 (width, height), None表示保持原图
        """
        self.data_path = Path(data_path)
        self.image_load_mode = image_load_mode
        self.max_workers = max_workers
        self.show_progress = show_progress and TQDM_AVAILABLE
        self.batch_size = batch_size
        self.image_size = self._normalize_image_size(image_size)
        self._raw_data: list[dict] = []
        self._image_cache: dict[str, PILImage.Image] = {}
        self._loaded_indices: set[int] = set()

        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        self._raw_data = self._load_jsonl()

        if self.show_progress:
            print(f"数据加载完成: {len(self._raw_data)} 条记录")
            with_images = sum(1 for d in self._raw_data if d.get("images"))
            print(f"含图片样本: {with_images} 条")
            if self.image_size is not None:
                print(f"训练图片尺寸: {self.image_size[0]}x{self.image_size[1]}")
        else:
            logger.info("数据加载完成: %d 条记录, %d 条含图片", len(self._raw_data), sum(1 for d in self._raw_data if d.get("images")))

        if self.image_load_mode == "preload" and self._has_images():
            self._preload_images()
        elif self.image_load_mode == "batch" and self._has_images():
            print(f"使用分批加载模式, 批次大小: {self.batch_size or 100}")

    def _load_jsonl(self) -> list[dict]:
        """加载 JSONL 数据文件"""
        data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    data.append(record)
                except json.JSONDecodeError as e:
                    logger.warning("跳过无效行: %s", e)
        return data

    def _has_images(self) -> bool:
        """检查数据集是否包含图片路径"""
        return any(len(sample.get("images", [])) > 0 for sample in self._raw_data)

    @staticmethod
    def _normalize_image_size(image_size: Optional[tuple[int, int]]) -> Optional[tuple[int, int]]:
        """标准化图片尺寸配置"""
        if image_size is None:
            return None
        if len(image_size) != 2:
            raise ValueError(f"image_size必须为(width, height), 当前: {image_size}")
        width, height = int(image_size[0]), int(image_size[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"image_size必须为正整数, 当前: {image_size}")
        return width, height

    @staticmethod
    def _get_resize_resample():
        """兼容不同 Pillow 版本的重采样枚举"""
        resampling = getattr(PILImage, "Resampling", None)
        if resampling is not None:
            return resampling.LANCZOS
        return PILImage.LANCZOS

    def _load_image_safe(self, image_path: str) -> Optional[PILImage.Image]:
        """安全加载单张图片，失败时返回 None

        注意: PIL.open() 使用延迟加载，只有访问像素数据时才真正加载
        """
        try:
            img = PILImage.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            if self.image_size is not None and img.size != self.image_size:
                img = img.resize(self.image_size, resample=self._get_resize_resample())
            return img
        except Exception as e:
            logger.warning("图片加载失败: %s, 错误: %s", image_path, e)
            return None

    def _preload_images(self) -> None:
        """多线程批量预加载所有图片到内存缓存

        使用 ThreadPoolExecutor 并行加载:
        - I/O 等待期间 GIL 自动释放，线程真正并行
        - 去重: 相同路径只加载一次
        - 进度条: 显示加载进度和内存状态
        """
        all_paths: set[str] = set()
        for sample in self._raw_data:
            for img_path in sample.get("images", []):
                all_paths.add(img_path)

        if not all_paths:
            return

        unique_paths = list(all_paths)
        total_count = len(unique_paths)

        if self.show_progress:
            print(f"开始预加载 {total_count} 张图片...")
            print_memory_status("加载前内存")
            pbar = create_progress_bar(
                total=total_count,
                desc="加载图片",
                unit="张",
            )
        else:
            logger.info("预加载 %d 张图片 (线程数: %d)...", total_count, self.max_workers)
            pbar = None

        success_count = 0
        fail_count = 0
        check_interval = max(1, total_count // 20)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._load_image_safe, p): p for p in unique_paths}
            for idx, future in enumerate(as_completed(futures)):
                img_path = futures[future]
                result = future.result()
                if result is not None:
                    self._image_cache[img_path] = result
                    success_count += 1
                else:
                    fail_count += 1

                if pbar:
                    pbar.update(1)
                    if idx % check_interval == 0 and idx > 0:
                        pbar.set_postfix(
                            {
                                "成功": success_count,
                                "失败": fail_count,
                                "缓存": len(self._image_cache),
                            }
                        )

        if pbar:
            pbar.close()
            print_memory_status("加载后内存")

        loaded_count = len(self._image_cache)
        if loaded_count < total_count:
            msg = f"图片加载完成: {loaded_count}/{total_count} 张成功 ({fail_count}张失败)"
            if self.show_progress:
                print(msg)
            else:
                logger.warning(msg)
        else:
            msg = f"图片预加载完成: {loaded_count} 张全部成功"
            if self.show_progress:
                print(msg)
            else:
                logger.info(msg)

    def preload_batch(self, start_idx: int, end_idx: int) -> None:
        """分批预加载指定范围的图片

        Args:
            start_idx: 起始样本索引
            end_idx: 结束样本索引 (不含)
        """
        paths_to_load: set[str] = set()
        for i in range(start_idx, min(end_idx, len(self._raw_data))):
            sample = self._raw_data[i]
            for img_path in sample.get("images", []):
                if img_path not in self._image_cache:
                    paths_to_load.add(img_path)

        if not paths_to_load:
            return

        path_list = list(paths_to_load)
        batch_count = len(path_list)

        desc = f"加载批次[{start_idx}-{end_idx}]"
        if self.show_progress:
            pbar = create_progress_bar(total=batch_count, desc=desc, unit="张")
        else:
            pbar = None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._load_image_safe, p): p for p in path_list}
            for future in as_completed(futures):
                img_path = futures[future]
                result = future.result()
                if result is not None:
                    self._image_cache[img_path] = result
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

    def clear_cache(self, keep_recent: int = 0) -> None:
        """清理图片缓存以释放内存

        Args:
            keep_recent: 保留最近访问的图片数量
        """
        if keep_recent > 0 and len(self._image_cache) > keep_recent:
            items = list(self._image_cache.items())
            self._image_cache = dict(items[-keep_recent:])
        else:
            self._image_cache.clear()

        gc.collect()
        if self.show_progress:
            print_memory_status("缓存清理后")

    def _get_pil_image(self, image_path: str) -> Optional[PILImage.Image]:
        """获取单个 PIL 图片对象

        根据加载模式决定从缓存读取还是从磁盘加载
        PIL.open() 使用延迟加载，只有访问像素时才真正加载到内存
        """
        if self.image_load_mode == "preload":
            return self._image_cache.get(image_path)
        elif self.image_load_mode == "batch":
            if image_path in self._image_cache:
                return self._image_cache[image_path]
            else:
                pil_img = self._load_image_safe(image_path)
                if pil_img is not None:
                    self._image_cache[image_path] = pil_img
                return pil_img
        else:
            return self._load_image_safe(image_path)

    def __len__(self) -> int:
        """数据集大小"""
        return len(self._raw_data)

    def __getitem__(self, idx: int) -> dict:
        """按索引获取处理后的单条数据

        返回官方格式: 图片嵌入在 messages content 中

        Returns:
            {"messages": [...]} (图片在 content 中，无单独 images 列)
        """
        if idx < 0 or idx >= len(self._raw_data):
            raise IndexError(f"索引超出范围: {idx} (有效范围: 0-{len(self._raw_data) - 1})")

        sample = self._raw_data[idx]
        messages = sample.get("messages", [])
        image_paths = sample.get("images", [])

        processed_messages = self._build_messages_with_images(messages, image_paths)

        return {"messages": processed_messages}

    def _build_messages_with_images(self, messages: list[dict], image_paths: list[str]) -> list[dict]:
        """构建包含图片的 messages（官方格式）

        图片直接嵌入在 user content 中，格式为:
        {"type": "image", "image": PIL.Image.Image}

        Args:
            messages: 原始消息列表
            image_paths: 图片路径列表

        Returns:
            处理后的消息列表（图片嵌入在 content 中）
        """
        processed = []
        image_inserted = False

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", [])

            if role == "user" and isinstance(content, list):
                new_content = []

                if not image_inserted and len(image_paths) > 0:
                    for img_path in image_paths:
                        pil_img = self._get_pil_image(img_path)
                        if pil_img is not None:
                            new_content.append({"type": "image", "image": pil_img})
                    image_inserted = True

                for item in content:
                    if item.get("type") == "image":
                        continue
                    new_content.append(item)

                processed.append({"role": role, "content": new_content})
            else:
                processed.append({"role": role, "content": content})

        return processed

    def to_conversation_list(
        self,
        show_memory_stats: bool = True,
        lazy_load: bool = True,
    ) -> list[dict]:
        """转换为官方格式的 conversation list

        这是 Unsloth 官方示例推荐的方式:
        - 返回 Python list of dicts
        - 图片嵌入在 messages content 中
        - 配合 UnslothVisionDataCollator 使用

        Args:
            show_memory_stats: 是否显示内存统计信息
            lazy_load: 是否延迟加载图片（只在训练时加载）

        Returns:
            list of dicts，格式为 [{"messages": [...]}]
        """
        total = len(self)

        if show_memory_stats and self.show_progress:
            print_memory_status("转换前内存")
            print(f"开始转换 {total} 条数据为 conversation list...")

        result = []

        if self.show_progress:
            pbar = create_progress_bar(
                total=total,
                desc="数据转换",
                unit="条",
            )
            check_interval = max(1, total // 20)
        else:
            pbar = None
            check_interval = 0

        for i in range(total):
            item = self[i]
            result.append(item)

            if pbar:
                pbar.update(1)
                if i % check_interval == 0 and i > 0:
                    mem = get_memory_usage()
                    if isinstance(mem["process_rss_gb"], float):
                        pbar.set_postfix({"RSS": f"{mem['process_rss_gb']}GB"})

        if pbar:
            pbar.close()

        if show_memory_stats and self.show_progress:
            print_memory_status("转换后内存")
            print(f"转换完成: {len(result)} 条数据")
            print("提示: 请配合 UnslothVisionDataCollator 使用此数据")

        return result

    def to_hf_dataset(
        self,
        remove_metadata: bool = True,
        show_memory_stats: bool = True,
    ) -> Dataset:
        """转换为 HuggingFace Dataset

        使用官方格式：图片嵌入在 messages content 中
        不再使用单独的 images 列，避免内存暴涨

        Args:
            remove_metadata: 是否移除 metadata 列 (训练不需要)
            show_memory_stats: 是否显示内存统计信息

        Returns:
            HuggingFace Dataset 对象
        """
        total = len(self)

        if show_memory_stats and self.show_progress:
            print_memory_status("转换前内存")
            print(f"开始转换 {total} 条数据...")

        all_messages = []

        if self.show_progress:
            pbar = create_progress_bar(
                total=total,
                desc="数据转换",
                unit="条",
            )
            check_interval = max(1, total // 20)
        else:
            pbar = None
            check_interval = 0

        for i in range(total):
            item = self[i]
            all_messages.append(item["messages"])

            if pbar:
                pbar.update(1)
                if i % check_interval == 0 and i > 0:
                    mem = get_memory_usage()
                    if isinstance(mem["process_rss_gb"], float):
                        pbar.set_postfix({"RSS": f"{mem['process_rss_gb']}GB"})

        if pbar:
            pbar.close()

        data_dict = {"messages": all_messages}
        ds = Dataset.from_dict(data_dict)

        if remove_metadata and "metadata" in ds.column_names:
            ds = ds.remove_columns(["metadata"])

        if show_memory_stats and self.show_progress:
            print_memory_status("转换后内存")
            print(f"转换完成: {len(ds)} 条数据")

        return ds

    def to_hf_dataset_batched(
        self,
        batch_size: int = 500,
        remove_metadata: bool = True,
        clear_cache_between_batches: bool = True,
        show_memory_stats: bool = True,
    ) -> Dataset:
        """分批转换为 HuggingFace Dataset，控制内存占用

        分批处理数据，每批处理后可选清理缓存，
        最后合并所有批次为完整的 Dataset。

        Args:
            batch_size: 每批处理的样本数量
            remove_metadata: 是否移除 metadata 列
            clear_cache_between_batches: 是否在批次间清理图片缓存
            show_memory_stats: 是否显示内存统计信息

        Returns:
            HuggingFace Dataset 对象
        """
        total = len(self)
        num_batches = (total + batch_size - 1) // batch_size

        if show_memory_stats and self.show_progress:
            print_memory_status("分批转换前内存")
            print(f"分批转换配置: 总样本={total}, 批次大小={batch_size}, 批次数={num_batches}")

        all_messages = []

        if self.show_progress:
            batch_pbar = create_progress_bar(
                total=num_batches,
                desc="批次处理",
                unit="批",
            )
        else:
            batch_pbar = None

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)

            if self.image_load_mode == "batch":
                self.preload_batch(start, end)

            batch_messages = []

            for i in range(start, end):
                item = self[i]
                batch_messages.append(item["messages"])

            all_messages.extend(batch_messages)

            if clear_cache_between_batches:
                keep_count = min(100, len(self._image_cache))
                self.clear_cache(keep_recent=keep_count)

            if batch_pbar:
                batch_pbar.update(1)
                mem = get_memory_usage()
                if isinstance(mem["process_rss_gb"], float):
                    batch_pbar.set_postfix(
                        {
                            "RSS": f"{mem['process_rss_gb']}GB",
                            "已完成": f"{end}/{total}",
                        }
                    )

            if show_memory_stats and self.show_progress and batch_idx % 5 == 0:
                print(f"  批次 {batch_idx + 1}/{num_batches}: {start}-{end} 条")

        if batch_pbar:
            batch_pbar.close()

        data_dict = {"messages": all_messages}
        ds = Dataset.from_dict(data_dict)

        if remove_metadata and "metadata" in ds.column_names:
            ds = ds.remove_columns(["metadata"])

        if show_memory_stats and self.show_progress:
            print_memory_status("分批转换完成")
            print(f"最终数据集: {len(ds)} 条")

        return ds

    def stats(self) -> dict:
        """返回数据集统计信息"""
        total = len(self._raw_data)
        with_images = sum(1 for d in self._raw_data if len(d.get("images", [])) > 0)
        total_images = sum(len(d.get("images", [])) for d in self._raw_data)
        cache_size = len(self._image_cache)

        stats_dict = {
            "total_samples": total,
            "samples_with_images": with_images,
            "samples_without_images": total - with_images,
            "total_image_paths": total_images,
            "cached_images": cache_size,
            "image_load_mode": self.image_load_mode,
            "unique_image_paths": len({p for d in self._raw_data for p in d.get("images", [])}),
            "configured_image_size": self.image_size,
        }

        mem = get_memory_usage()
        stats_dict["memory_rss_gb"] = mem["process_rss_gb"]
        stats_dict["memory_available_gb"] = mem["system_available_gb"]

        return stats_dict


def create_multimodal_dataset(
    data_path: str,
    image_load_mode: str = "lazy",
    max_workers: int = 8,
    show_progress: bool = True,
    batch_size: Optional[int] = None,
    image_size: Optional[tuple[int, int]] = None,
) -> MultimodalDataset:
    """便捷函数: 创建多模态数据集

    Args:
        data_path: JSONL 数据文件路径
        image_load_mode: 图片加载策略 ("preload", "lazy", "batch")
        max_workers: 程池大小 (仅 preload/batch 模式)
        show_progress: 是否显示进度条
        batch_size: 分批加载时的批次大小 (仅 batch 模式)
        image_size: 训练前统一重采样尺寸 (width, height)

    Returns:
        MultimodalDataset 宝例
    """
    return MultimodalDataset(
        data_path=data_path,
        image_load_mode=image_load_mode,
        max_workers=max_workers,
        show_progress=show_progress,
        batch_size=batch_size,
        image_size=image_size,
    )


def create_vision_dataset(
    data_path: str,
    max_workers: int = 8,
    show_progress: bool = True,
    use_batched: bool = False,
    batch_size: int = 500,
    return_list: bool = True,
    image_size: Optional[tuple[int, int]] = None,
) -> Union[list[dict], Dataset]:
    """便捷函数: 创建视觉微调数据集

    推荐使用 return_list=True 配合 UnslothVisionDataCollator:
        dataset = create_vision_dataset("train.jsonl", return_list=True)
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            data_collator=UnslothVisionDataCollator(model, processor),
            ...
        )

    Args:
        data_path: JSONL 数据文件路径
        max_workers: 图片预加载线程池大小
        show_progress: 是否显示进度条
        use_batched: 是否使用分批处理模式 (推荐大数据集)
        batch_size: 分批处理时的批次大小
        return_list: 是否返回 Python list (官方推荐) 而非 HuggingFace Dataset
        image_size: 训练前统一重采样尺寸 (width, height)

    Returns:
        处理完成的数据集（list 或 Dataset）
    """
    mm_ds = MultimodalDataset(
        data_path=data_path,
        image_load_mode="lazy",
        max_workers=max_workers,
        show_progress=show_progress,
        batch_size=batch_size if use_batched else None,
        image_size=image_size,
    )

    if return_list:
        return mm_ds.to_conversation_list()
    elif use_batched:
        return mm_ds.to_hf_dataset_batched(batch_size=batch_size)
    else:
        return mm_ds.to_hf_dataset()

