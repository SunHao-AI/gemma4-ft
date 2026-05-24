from __future__ import annotations

# Unsloth 必须在所有其他导入之前导入以确保优化生效
import unsloth  # noqa: F401

import gc
import json
import os
import platform
import re
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import requests
import torch
from PIL import Image, ImageDraw, ImageFont

from unsloth_finetune.training.distributed.adapter_utils import prepared_adapter_dir


from unsloth_finetune.data.labelme.detection_format import (
    DetectionPromptBuilder,
    build_cn_normalized_detection_prompt,
    build_en_normalized_detection_prompt,
    convert_xyxy_to_format,
)


_FONT_CACHE_DIR = Path.home() / ".cache" / "unsloth_fonts"
_NOTO_SANS_SC_URL = (
    "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/"
    "NotoSansCJKsc-Regular.otf"
)
_NOTO_SANS_SC_FILE = "NotoSansCJKsc-Regular.otf"


def _download_font(url: str, dest: Path, timeout: int = 30) -> bool:
    """下载字体文件到本地缓存目录, 返回是否成功"""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=timeout, stream=True)
        if resp.status_code != 200:
            return False
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception:
        return False


def configure_matplotlib_for_chinese(
    plt_module,
    font_manager=None,
    warn: bool = False,
    auto_download: bool = True,
) -> Optional[str]:
    chinese_fonts = [
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        # 下载后的字体名称 (Noto CJK OTF 注册名)
        "Noto Sans CJK SC",
        "NotoSansCJKsc-Regular",
    ]
    available_fonts = []
    if font_manager is not None:
        available_fonts = [font.name for font in font_manager.fontManager.ttflist]

    font_found = None
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            font_found = font_name
            break

    # 系统中未找到中文字体 → 尝试自动下载 Noto Sans CJK SC
    if font_found is None and auto_download:
        cached_font = _FONT_CACHE_DIR / _NOTO_SANS_SC_FILE
        need_download = not cached_font.exists()
        if need_download:
            if warn:
                print("未找到系统中文字体, 正在尝试下载 Noto Sans CJK SC ...")
            downloaded = _download_font(_NOTO_SANS_SC_URL, cached_font)
            if downloaded and warn:
                print(f"字体下载成功: {cached_font}")
            elif not downloaded and warn:
                print("字体下载失败, 将回退使用英文字体 (DejaVu Sans)")

        # 无论刚下载还是已有缓存, 尝试注册
        if cached_font.exists():
            try:
                if font_manager is not None:
                    font_manager.fontManager.addfont(str(cached_font))
                    # 重新扫描以获取新注册的字体名
                    available_fonts = [
                        font.name for font in font_manager.fontManager.ttflist
                    ]
                    for font_name in chinese_fonts:
                        if font_name in available_fonts:
                            font_found = font_name
                            break
                # 也直接尝试用路径注册
                if font_found is None:
                    import matplotlib.font_manager as _fm
                    _fm.fontManager.addfont(str(cached_font))
                    for fn in chinese_fonts:
                        if fn in [f.name for f in _fm.fontManager.ttflist]:
                            font_found = fn
                            break
            except Exception:
                pass

    if font_found:
        plt_module.rcParams["font.sans-serif"] = [font_found, "DejaVu Sans"]
        plt_module.rcParams["axes.unicode_minus"] = False
        return font_found

    plt_module.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    if warn:
        print("警告: 未找到中文字体, 图表中文可能显示为方块")
    return None


def print_torch_runtime_info(torch_module: Any, version_label: str = "PyTorch 版本") -> None:
    print(f"{version_label}: {torch_module.__version__}")
    if torch_module.cuda.is_available():
        gpu_name = torch_module.cuda.get_device_name(0)
        gpu_memory = torch_module.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_memory:.1f} GB")
        print(f"CUDA版本: {torch_module.version.cuda}")
    else:
        print("警告: 未检测到 GPU")


class ModelLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
        self._is_loaded = False

    def _patch_peft_for_gemma4(self) -> bool:
        try:
            from peft.tuners.lora import model as lora_model

            original = lora_model.LoraModel._create_new_module

            def patched(lora_config, adapter_name, target, **kwargs):
                if target.__class__.__name__ == "Gemma4ClippableLinear" and hasattr(target, "linear"):
                    return original(lora_config, adapter_name, target.linear, **kwargs)
                return original(lora_config, adapter_name, target, **kwargs)

            lora_model.LoraModel._create_new_module = staticmethod(patched)
            print("PEFT已patch，支持Gemma4ClippableLinear")
            return True
        except Exception as exc:
            print(f"Patch失败: {exc}")
            return False

    def load_model(self) -> bool:
        try:
            print(f"正在加载模型: {self.config.get('name', 'Unknown')}")
            os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
            if not self.config.get("enable_compile", False):
                os.environ["TORCH_COMPILE_DISABLE"] = "1"
                try:
                    import torch._dynamo

                    torch._dynamo.config.suppress_errors = True
                    torch._dynamo.reset()
                except Exception:
                    pass

            from unsloth import FastVisionModel

            from_pretrained_kwargs = {
                "model_name": self.config["base_model_path"],
                "max_seq_length": self.config["max_seq_length"],
                "load_in_4bit": self.config["load_in_4bit"],
                "device_map": self.config.get("device_map"),
                "disable_log_stats": True,
            }
            attn_impl = self.config.get("attn_implementation")
            if attn_impl is not None:
                from_pretrained_kwargs["attn_implementation"] = attn_impl

            self.model, self.processor = FastVisionModel.from_pretrained(**from_pretrained_kwargs)

            lora_path = self.config.get("lora_adapter_path")
            if lora_path and os.path.exists(lora_path):
                print(f"正在加载 LoRA adapter: {lora_path}")
                self._patch_peft_for_gemma4()
                from peft import PeftModel

                with prepared_adapter_dir(lora_path) as prepared_lora_path:
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        str(prepared_lora_path),
                        is_trainable=False,
                    )
                print("LoRA adapter 加载成功")
            elif lora_path:
                print(f"LoRA adapter 路径不存在: {lora_path}")
                print("将使用基础模型进行推理")

            self._is_loaded = True
            return True
        except Exception as exc:
            print(f"模型加载失败: {exc}")
            traceback.print_exc()
            return False

    def unload_model(self) -> None:
        if not self._is_loaded:
            return

        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_model_info(self) -> Dict[str, Any]:
        if not self._is_loaded:
            return {"status": "未加载"}

        total_params = sum(parameter.numel() for parameter in self.model.parameters())
        return {
            "status": "已加载",
            "total_params": total_params,
            "params_billion": total_params / 1e9,
        }


class ImageLoader:
    SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]

    def __init__(self):
        self.image: Optional[Image.Image] = None
        self.image_path: Optional[str] = None
        self._width = 0
        self._height = 0

    def load_from_local(self, path: str) -> bool:
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                raise FileNotFoundError(f"文件不存在: {path}")
            if path_obj.suffix.lower() not in self.SUPPORTED_FORMATS:
                raise ValueError(f"不支持的图像格式: {path_obj.suffix}")

            self.image = Image.open(path_obj).convert("RGB")
            self.image_path = path
            self._width, self._height = self.image.size
            print(f"图像加载成功: {path}")
            print(f"尺寸: {self._width} x {self._height}")
            return True
        except Exception as exc:
            print(f"本地图像加载失败: {exc}")
            return False

    def load_from_url(self, url: str, timeout: int = 30) -> bool:
        try:
            print(f"正在从 URL 下载图像: {url}")
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "image" not in content_type:
                raise ValueError(f"URL 不是图像类型: {content_type}")

            self.image = Image.open(BytesIO(response.content)).convert("RGB")
            self.image_path = url
            self._width, self._height = self.image.size
            print("图像下载成功")
            print(f"尺寸: {self._width} x {self._height}")
            return True
        except requests.exceptions.Timeout:
            print(f"请求超时 ({timeout}秒)")
            return False
        except requests.exceptions.RequestException as exc:
            print(f"网络请求失败: {exc}")
            return False
        except Exception as exc:
            print(f"URL 图像加载失败: {exc}")
            return False

    def load(self, source: str) -> bool:
        if source.startswith(("http://", "https://")):
            return self.load_from_url(source)
        return self.load_from_local(source)

    def get_image(self) -> Optional[Image.Image]:
        return self.image

    def get_size(self) -> Tuple[int, int]:
        return self._width, self._height

    def display(self, plt_module, title: str = "图像") -> None:
        if self.image is None:
            print("未加载图像")
            return

        plt_module.figure(figsize=(10, 8))
        plt_module.imshow(self.image)
        plt_module.title(title)
        plt_module.axis("off")
        plt_module.tight_layout()
        plt_module.show()


class ObjectDetector:
    def __init__(
        self,
        model_loader: ModelLoader,
        prompt_builder: DetectionPromptBuilder = build_en_normalized_detection_prompt,
        temperature: float = 0.7,
        top_p: float = 0.9,
        coord_format: str = "xyxy",
        coord_norm: str = "auto",
    ):
        self.model_loader = model_loader
        self.prompt_builder = prompt_builder
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.coord_format = coord_format
        self.coord_norm = coord_norm

    def _resolve_model_device(self, model) -> torch.device:
        model_device = getattr(model, "device", None)
        if model_device is not None:
            return model_device
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _move_inputs_to_device(self, inputs, device: torch.device):
        if hasattr(inputs, "to"):
            return inputs.to(device)
        if isinstance(inputs, dict):
            moved = {}
            for key, value in inputs.items():
                moved[key] = value.to(device) if hasattr(value, "to") else value
            return moved
        raise TypeError(f"Unsupported processor output type: {type(inputs)!r}")

    def _prepare_generation_inputs(self, images: Sequence[Image.Image], queries: Sequence[str], padding: bool):
        model = self.model_loader.model
        processor = self.model_loader.processor
        prompts = [self.prompt_builder(query) for query in queries]
        messages_batch = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for image, prompt in zip(images, prompts)
        ]
        texts = [
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]

        processor_kwargs = {
            "text": texts,
            "images": list(images),
            "return_tensors": "pt",
        }
        if padding:
            processor_kwargs["padding"] = True

        inputs = processor(**processor_kwargs)
        return self._move_inputs_to_device(inputs, self._resolve_model_device(model))

    def _decode_generated_responses(self, processor, inputs, outputs) -> List[str]:
        if "attention_mask" in inputs:
            prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        elif "input_ids" in inputs:
            prompt_lengths = [inputs["input_ids"].shape[1]] * inputs["input_ids"].shape[0]
        else:
            prompt_lengths = [0] * outputs.shape[0]

        generated_only = []
        for index, output in enumerate(outputs):
            start = int(prompt_lengths[index]) if index < len(prompt_lengths) else 0
            generated_only.append(output[start:])
        return processor.batch_decode(generated_only, skip_special_tokens=True)

    def detect(self, image: Image.Image, query: str, max_new_tokens: int = 512) -> Dict[str, Any]:
        if not self.model_loader.is_loaded():
            return {"error": "模型未加载", "success": False}

        try:
            model = self.model_loader.model
            processor = self.model_loader.processor
            inputs = self._prepare_generation_inputs([image], [query], padding=False)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                )

            response = self._decode_generated_responses(processor, inputs, outputs)[0]
            width, height = image.size
            detections = self.parse_response(
                response, width, height,
                coord_format=self.coord_format,
                coord_norm=self.coord_norm,
            )
            return {
                "success": True,
                "raw_response": response,
                "detections": detections,
                "query": query,
            }
        except Exception as exc:
            return {"error": str(exc), "success": False}

    def detect_batch(
        self,
        images: Sequence[Image.Image],
        queries: Sequence[str],
        max_new_tokens: int = 512,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.model_loader.is_loaded():
            return [{"error": "模型未加载", "success": False} for _ in images]
        if len(images) != len(queries):
            raise ValueError(f"images and queries length mismatch: {len(images)} != {len(queries)}")
        if not images:
            return []

        model = self.model_loader.model
        processor = self.model_loader.processor
        batch_size = batch_size or len(images)
        results: List[Dict[str, Any]] = []
        for start in range(0, len(images), batch_size):
            image_batch = list(images[start : start + batch_size])
            query_batch = list(queries[start : start + batch_size])
            try:
                inputs = self._prepare_generation_inputs(image_batch, query_batch, padding=len(image_batch) > 1)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                    )

                responses = self._decode_generated_responses(processor, inputs, outputs)
                for image, query, response in zip(image_batch, query_batch, responses):
                    width, height = image.size
                    detections = self.parse_response(
                        response, width, height,
                        coord_format=self.coord_format,
                        coord_norm=self.coord_norm,
                    )
                    results.append(
                        {
                            "success": True,
                            "raw_response": response,
                            "detections": detections,
                            "query": query,
                        }
                    )
            except Exception as batch_error:
                for image, query in zip(image_batch, query_batch):
                    single = self.detect(image, query, max_new_tokens=max_new_tokens)
                    if not single.get("success"):
                        single["error"] = single.get("error") or str(batch_error)
                    results.append(single)
        return results

    @staticmethod
    def parse_response(
        response: str,
        width: int,
        height: int,
        coord_format: str = "xyxy",
        coord_norm: str = "auto",
    ) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []

        def _is_normalized(coords: list) -> bool:
            return all(0 <= v <= 1 for v in coords)

        def _is_1000_based(coords: list) -> bool:
            return all(0 <= v <= 1000 for v in coords) and not _is_normalized(coords)

        def _effective_norm(coords: list) -> str:
            if coord_norm != "auto":
                return coord_norm
            if _is_normalized(coords):
                return "norm_1"
            if _is_1000_based(coords):
                return "norm_1000"
            return "raw"

        def convert_coords(coords: list) -> Tuple[int, int, int, int]:
            eff = _effective_norm(coords)
            if eff == "norm_1":
                x1 = int(coords[0] * width)
                y1 = int(coords[1] * height)
                x2 = int(coords[2] * width)
                y2 = int(coords[3] * height)
            elif eff == "norm_1000":
                x1 = int(coords[0] / 1000 * width)
                y1 = int(coords[1] / 1000 * height)
                x2 = int(coords[2] / 1000 * width)
                y2 = int(coords[3] / 1000 * height)
            else:  # raw pixel coords
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            return x1, y1, x2, y2

        def sanitize_box(x1: int, y1: int, x2: int, y2: int):
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            if x2 <= x1 or y2 <= y1:
                return None
            return x1, y1, x2, y2

        def extract_json_array(text: str) -> Optional[str]:
            json_block = re.search(r"```json\s*([\s\S]*?)\s*```", text)
            if json_block:
                return json_block.group(1).strip()

            start_idx = text.find("[")
            if start_idx == -1:
                return None

            bracket_count = 0
            for index, character in enumerate(text[start_idx:], start_idx):
                if character == "[":
                    bracket_count += 1
                elif character == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        return text[start_idx : index + 1]
            return None

        def append_detection(item: dict):
            coords = item.get("box_2d")
            if not isinstance(coords, list) or len(coords) != 4:
                return

            x1, y1, x2, y2 = convert_coords(coords)
            sanitized = sanitize_box(x1, y1, x2, y2)
            if sanitized is None:
                return

            confidence = item.get("confidence", 0.85)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.85

            detections.append(
                {
                    "bbox": [sanitized[0], sanitized[1], sanitized[2], sanitized[3]],
                    "bbox_out": convert_xyxy_to_format(
                        sanitized[0], sanitized[1], sanitized[2], sanitized[3], coord_format
                    ),
                    "label": item.get("label", "object"),
                    "confidence": max(0.0, min(confidence, 1.0)),
                }
            )

        json_str = extract_json_array(response)
        if json_str:
            try:
                json_data = json.loads(json_str)
                if isinstance(json_data, list):
                    for item in json_data:
                        if isinstance(item, dict):
                            append_detection(item)
                if detections:
                    return detections
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        obj_pattern = r'\{[^{}]*"box_2d"[^{}]*\}'
        for obj_str in re.findall(obj_pattern, response, re.DOTALL):
            try:
                obj = json.loads(obj_str)
                if isinstance(obj, dict):
                    append_detection(obj)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        return detections


class DetectionVisualizer:
    DEFAULT_COLORS = [
        "#FF3838",
        "#FF9D00",
        "#FF701F",
        "#FFB21D",
        "#CFD231",
        "#48F90A",
        "#92CC17",
        "#3DDB86",
        "#1A9F34",
        "#00D4BB",
        "#2C99A8",
        "#00C2FF",
        "#34459D",
        "#6473E9",
        "#0018EC",
        "#8438FF",
        "#520085",
        "#CFD894",
        "#FF78A5",
        "#FF3838",
    ]

    def __init__(self, colors: Optional[Sequence[str]] = None, font_size: int = 20):
        self.colors = list(colors or self.DEFAULT_COLORS)
        self.font_size = font_size
        self.font = self._load_font()

    def _load_font(self):
        font_candidates = [
            ("Microsoft YaHei", "msyh.ttc"),
            ("SimHei", "simhei.ttf"),
            ("SimSun", "simsun.ttc"),
            ("Arial", "arial.ttf"),
        ]

        font_dir = None
        system = platform.system()
        if system == "Windows":
            font_dir = Path("C:/Windows/Fonts")
        elif system == "Linux":
            font_dir = Path("/usr/share/fonts")

        for font_name, font_file in font_candidates:
            try:
                return ImageFont.truetype(font_name, self.font_size)
            except Exception:
                pass

            if font_dir and font_dir.exists():
                font_path = font_dir / font_file
                if font_path.exists():
                    try:
                        return ImageFont.truetype(str(font_path), self.font_size)
                    except Exception:
                        pass

        # 系统字体均不可用 → 尝试使用下载缓存的 Noto Sans CJK SC
        cached_font = _FONT_CACHE_DIR / _NOTO_SANS_SC_FILE
        if cached_font.exists():
            try:
                return ImageFont.truetype(str(cached_font), self.font_size)
            except Exception:
                pass

        try:
            return ImageFont.load_default(size=self.font_size)
        except TypeError:
            return ImageFont.load_default()

    def _get_color(self, index: int) -> str:
        return self.colors[index % len(self.colors)]

    def draw_detections(
        self,
        image: Image.Image,
        detections: Sequence[Dict[str, Any]],
        box_width: int = 3,
        show_confidence: bool = True,
    ) -> Image.Image:
        if not detections:
            return image

        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)

        for index, detection in enumerate(detections):
            bbox = detection.get("bbox", [0, 0, 0, 0])
            label = detection.get("label", "未知")
            confidence = detection.get("confidence", 0)

            if len(bbox) != 4 or not all(isinstance(value, (int, float)) for value in bbox):
                continue

            x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            color = self._get_color(index)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

            text = f"{label} {confidence:.0%}" if show_confidence else str(label)
            text_bbox = draw.textbbox((x1, y1), text, font=self.font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            fill_y1 = y1 - text_height - 4
            if fill_y1 < 0:
                fill_y1 = y1 + 4
            fill_y2 = fill_y1 + text_height + 2

            draw.rectangle([x1, fill_y1, x1 + text_width + 4, fill_y2], fill=color)
            draw.text((x1 + 2, fill_y1 + 1), text, fill="white", font=self.font)

        return img_draw

    def display_result(
        self,
        plt_module,
        original: Image.Image,
        result: Image.Image,
        detections: Sequence[Dict[str, Any]],
        figsize: Tuple[int, int] = (15, 8),
    ) -> None:
        fig, axes = plt_module.subplots(1, 2, figsize=figsize)
        axes[0].imshow(original)
        axes[0].set_title("原始图像")
        axes[0].axis("off")

        axes[1].imshow(result)
        axes[1].set_title(f"检测结果 ({len(detections)} 个目标)")
        axes[1].axis("off")

        plt_module.tight_layout()
        plt_module.show()

        if detections:
            print("\n检测结果详情:")
            for index, detection in enumerate(detections, start=1):
                bbox = detection.get("bbox", [0, 0, 0, 0])
                print(f"  {index}. {detection.get('label', '未知')} - 置信度 {detection.get('confidence', 0):.0%}")
                print(f"     边界框: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
        else:
            print("\n未检测到目标")

    def save_result(self, image: Image.Image, output_path: str) -> bool:
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            image.save(output_path, quality=95)
            print(f"结果已保存: {output_path}")
            return True
        except Exception as exc:
            print(f"保存失败: {exc}")
            return False


class ComparisonVisualizer(DetectionVisualizer):
    def create_comparison_plot(
        self,
        plt_module,
        grid_spec,
        image: Image.Image,
        det_base: Sequence[Dict[str, Any]],
        det_finetuned: Sequence[Dict[str, Any]],
        iou_stats: Dict[str, Any],
    ) -> None:
        fig = plt_module.figure(figsize=(16, 10))
        gs = grid_spec(2, 2, figure=fig, height_ratios=[3, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        ax1.imshow(self.draw_detections(image, det_base))
        ax1.set_title(f"原始模型\n检测数量: {len(det_base)}", fontsize=12, fontweight="bold")
        ax1.axis("off")

        ax2.imshow(self.draw_detections(image, det_finetuned))
        ax2.set_title(f"微调模型\n检测数量: {len(det_finetuned)}", fontsize=12, fontweight="bold")
        ax2.axis("off")

        stats_text = "IOU统计:\n"
        stats_text += f"平均IOU: {iou_stats.get('mean_iou', 0):.3f}\n"
        stats_text += f"最大IOU: {iou_stats.get('max_iou', 0):.3f}\n"
        stats_text += f"匹配数量: {iou_stats.get('num_pairs', 0)}\n"
        ax3.text(0.5, 0.5, stats_text, ha="center", va="center", fontsize=11, family="monospace")
        ax3.axis("off")

        plt_module.tight_layout()
        plt_module.show()


class ObjectDetectionPipeline:
    def __init__(
        self,
        model_config: Dict[str, Any],
        prompt_builder: DetectionPromptBuilder = build_cn_normalized_detection_prompt,
        temperature: float = 0.7,
        top_p: float = 0.9,
        coord_format: str = "xyxy",
        coord_norm: str = "auto",
    ):
        self.model_loader = ModelLoader(model_config)
        self.image_loader = ImageLoader()
        self.visualizer = DetectionVisualizer()
        self.detector = ObjectDetector(
            self.model_loader,
            prompt_builder=prompt_builder,
            temperature=temperature,
            top_p=top_p,
            coord_format=coord_format,
            coord_norm=coord_norm,
        )
        self._initialized = False

    def initialize(self) -> bool:
        success = self.model_loader.load_model()
        self._initialized = success
        return success

    def run_detection(
        self,
        plt_module,
        image_source: str,
        query: str,
        output_path: Optional[str] = None,
        display_result: bool = True,
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        if not self._initialized:
            return {"error": "流程未初始化，请先调用 initialize()", "success": False}

        result: Dict[str, Any] = {
            "image_source": image_source,
            "query": query,
            "success": False,
        }

        if not self.image_loader.load(image_source):
            result["error"] = "图像加载失败"
            return result

        original_image = self.image_loader.get_image()
        result["image_size"] = self.image_loader.get_size()

        detection_result = self.detector.detect(original_image, query, max_new_tokens=max_new_tokens)
        if not detection_result.get("success", False):
            result["error"] = detection_result.get("error", "检测失败")
            return result

        detections = detection_result.get("detections", [])
        result["raw_response"] = detection_result.get("raw_response", "")
        print(f"\n[DEBUG] 模型原始响应:\n{result['raw_response']}\n")
        print(f"[DEBUG] coord_format={self.detector.coord_format}, coord_norm={self.detector.coord_norm}")
        if detections:
            raw_coords = []
            for d in detection_result.get("detections", []):
                bbox = d.get("bbox", [0, 0, 0, 0])
                raw_coords.append(f"  像素坐标: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            print("[DEBUG] 解析结果:")
            print("\n".join(raw_coords))
        result["detections"] = detections
        result["num_detections"] = len(detections)

        result_image = self.visualizer.draw_detections(original_image, detections)
        result["result_image"] = result_image

        if output_path:
            self.visualizer.save_result(result_image, output_path)
            result["output_path"] = output_path

        if display_result:
            if detections:
                self.visualizer.display_result(plt_module, original_image, result_image, detections)
            else:
                print("\n未检测到指定目标")
                print(f"模型响应: {result.get('raw_response', '')}")
                self.image_loader.display(plt_module, "原始图像 - 无检测结果")

        result["success"] = True
        return result

