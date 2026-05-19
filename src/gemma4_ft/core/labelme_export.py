import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _resolve_image_size(
    result: dict,
    image_loader: Optional[Callable[[str], Any]] = None,
) -> tuple[int, int]:
    width = result.get("image_width")
    height = result.get("image_height")
    if isinstance(width, (int, float)) and isinstance(height, (int, float)) and width > 0 and height > 0:
        return int(width), int(height)

    if image_loader is not None:
        image = image_loader(result["image_path"])
        if image is not None:
            loaded_width, loaded_height = image.size
            return int(loaded_width), int(loaded_height)

    metadata = result.get("metadata", {})
    fallback_width = metadata.get("image_width", 1000)
    fallback_height = metadata.get("image_height", 1000)
    return int(fallback_width), int(fallback_height)


def build_labelme_payload(
    result: dict,
    image_loader: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    width, height = _resolve_image_size(result, image_loader=image_loader)

    shapes = []
    for det in result.get("detections", []):
        bbox = det.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        shapes.append(
            {
                "label": det.get("label", "object"),
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "group_id": None,
                "description": (
                    f"confidence={det.get('confidence', 0.0):.4f},"
                    f"gpu={result.get('gpu_id')},model={result.get('model_type')}"
                ),
                "shape_type": "rectangle",
                "flags": {},
            }
        )

    return {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": Path(result["image_path"]).name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "metadata": {
            "gpu_id": result.get("gpu_id"),
            "model_type": result.get("model_type"),
            "precision": result.get("metrics", {}).get("precision", 0.0),
            "recall": result.get("metrics", {}).get("recall", 0.0),
            "f1": result.get("metrics", {}).get("f1", 0.0),
            "num_detections": len(result.get("detections", [])),
        },
        "raw_inference_data": {
            "query": result.get("query", ""),
            "raw_response": result.get("raw_response", ""),
            "detections_count": len(result.get("detections", [])),
            "parse_success": len(result.get("detections", [])) > 0,
            "inference_success": result.get("inference_success", True),
            "inference_error": result.get("inference_error", ""),
        },
    }


def build_labelme_output_path(result: dict, labelme_dir: Path) -> Path:
    image_path = Path(result["image_path"])
    stem = image_path.stem or "image"
    index = result.get("index")
    if index is None:
        return labelme_dir / f"{stem}.json"
    return labelme_dir / f"{int(index):06d}_{stem}.json"


def save_labelme_results(
    results: List[dict],
    labelme_dir: Path,
    image_loader: Optional[Callable[[str], Any]] = None,
    progress_factory: Optional[Callable[..., Any]] = None,
    progress_desc: str = "LabelMe export",
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    labelme_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    failed_items = []
    progress_bar = None
    if progress_factory is not None and results:
        progress_bar = progress_factory(total=len(results), desc=progress_desc, unit="file")

    try:
        for result in results:
            try:
                output_path = build_labelme_output_path(result, labelme_dir)
                payload = build_labelme_payload(result, image_loader=image_loader)
                with open(output_path, "w", encoding="utf-8") as file_obj:
                    json.dump(payload, file_obj, ensure_ascii=False, indent=2)
                written += 1
            except Exception as exc:
                image_path = result.get("image_path", "<unknown>")
                failed_items.append({"image_path": image_path, "error": str(exc)})
                if log_fn is not None:
                    log_fn(f"LabelMe export failed for {image_path}: {exc}")
            finally:
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix(written=written, failed=len(failed_items), refresh=False)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return {
        "written": written,
        "failed": len(failed_items),
        "failed_items": failed_items,
    }
