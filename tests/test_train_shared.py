from unsloth_finetune.notebooking.train_shared import discover_eval_data_path


def test_discover_eval_data_path_prefers_test_then_valid_then_val(tmp_path):
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{}", encoding="utf-8")

    test_path = tmp_path / "test.jsonl"
    test_path.write_text("{}", encoding="utf-8")
    valid_path = tmp_path / "valid.jsonl"
    valid_path.write_text("{}", encoding="utf-8")
    val_path = tmp_path / "val.jsonl"
    val_path.write_text("{}", encoding="utf-8")

    path, split = discover_eval_data_path(train_path)

    assert path == str(test_path.resolve())
    assert split == "test"


def test_discover_eval_data_path_supports_val_name(tmp_path):
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{}", encoding="utf-8")
    val_path = tmp_path / "val.jsonl"
    val_path.write_text("{}", encoding="utf-8")

    path, split = discover_eval_data_path(train_path)

    assert path == str(val_path.resolve())
    assert split == "val"
