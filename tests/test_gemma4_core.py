import os
import sys
from pathlib import Path

from gemma4_core.bootstrap import bootstrap_notebook_context
from gemma4_core.labelme_export import (
    build_labelme_output_path,
    build_labelme_payload,
    save_labelme_results,
)


class _FakeImage:
    def __init__(self, width: int, height: int):
        self.size = (width, height)


def test_bootstrap_notebook_context_resolves_project_root(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    notebook_dir = project_root / "notebooks"
    notebook_dir.mkdir(parents=True)
    notebook_file = notebook_dir / "demo.ipynb"
    notebook_file.write_text("{}", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")

    monkeypatch.delenv("GEMMA4_NOTEBOOK_DIR", raising=False)
    monkeypatch.delenv("GEMMA4_NOTEBOOK_FILE", raising=False)
    original_sys_path = list(sys.path)

    try:
        context = bootstrap_notebook_context(notebook_file=str(notebook_file), cwd=project_root)
    finally:
        sys.path[:] = original_sys_path

    assert context["NOTEBOOK_DIR"] == notebook_dir.resolve()
    assert context["PROJECT_ROOT"] == project_root.resolve()
    assert os.environ["GEMMA4_NOTEBOOK_DIR"] == str(notebook_dir.resolve())
    assert os.environ["GEMMA4_NOTEBOOK_FILE"] == str(notebook_file.resolve())


def test_build_labelme_payload_prefers_result_image_size():
    result = {
        "image_path": "demo.png",
        "image_width": 640,
        "image_height": 480,
        "gpu_id": 0,
        "model_type": "finetuned",
        "detections": [{"bbox": [10, 20, 30, 40], "label": "target", "confidence": 0.9}],
        "metrics": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "query": "find target",
        "raw_response": "[]",
        "inference_success": True,
        "inference_error": "",
    }

    payload = build_labelme_payload(result)

    assert payload["imageWidth"] == 640
    assert payload["imageHeight"] == 480
    assert payload["shapes"][0]["label"] == "target"


def test_save_labelme_results_writes_files_and_continues_after_failures(tmp_path):
    results = [
        {
            "index": 1,
            "image_path": str(tmp_path / "ok.png"),
            "image_width": 800,
            "image_height": 600,
            "gpu_id": 0,
            "model_type": "base",
            "detections": [{"bbox": [1, 2, 3, 4], "label": "box", "confidence": 0.8}],
            "metrics": {},
            "query": "",
            "raw_response": "",
            "inference_success": True,
            "inference_error": "",
        },
        {
            "index": 2,
            # image_path missing on purpose to force output-path construction failure
            "gpu_id": 0,
            "model_type": "base",
            "detections": [],
            "metrics": {},
            "query": "",
            "raw_response": "",
            "inference_success": False,
            "inference_error": "missing image",
        },
    ]

    messages = []
    summary = save_labelme_results(
        results,
        tmp_path / "labelme",
        image_loader=lambda _: _FakeImage(800, 600),
        log_fn=messages.append,
    )

    expected_path = build_labelme_output_path(results[0], tmp_path / "labelme")
    assert expected_path.exists()
    assert summary["written"] == 1
    assert summary["failed"] == 1
    assert messages
