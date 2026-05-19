import json

from unsloth_finetune.notebooking.data_prep_shared import prescan_label_statistics


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_prescan_label_statistics_is_deterministic_across_worker_counts(tmp_path, monkeypatch):
    monkeypatch.setattr("unsloth_finetune.notebooking.data_prep_shared.create_progress_bar", lambda **_: None)

    _write_json(
        tmp_path / "remote.json",
        {
            "imageUrl": "https://example.com/cat.jpg",
            "shapes": [{"label": "cat"}, {"label": "dog"}],
        },
    )
    _write_json(
        tmp_path / "local.json",
        {
            "imagePath": "local.jpg",
            "shapes": [{"label": "cat"}, {"points": []}],
        },
    )
    (tmp_path / "broken.json").write_text("{not-valid-json", encoding="utf-8")
    _write_json(
        tmp_path / "no_image.json",
        {
            "shapes": [{"label": "dog"}],
        },
    )
    _write_json(
        tmp_path / "remote_with_path.json",
        {
            "imageUrl": "https://example.com/dual.jpg",
            "imagePath": "dual.jpg",
            "shapes": [],
        },
    )

    single_worker_result = prescan_label_statistics(tmp_path, max_workers=1)
    multi_worker_result = prescan_label_statistics(tmp_path, max_workers=4)

    assert multi_worker_result == single_worker_result
    assert multi_worker_result == {
        "total_json_files": 5,
        "scanned_files": 4,
        "skipped_files": 1,
        "image_url_files": 2,
        "local_image_files": 1,
        "label_counts": {"cat": 2, "dog": 2, "unknown": 1},
        "total_labels": 3,
        "total_instances": 5,
    }
    assert multi_worker_result["scanned_files"] + multi_worker_result["skipped_files"] == multi_worker_result["total_json_files"]
