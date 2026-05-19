"""
测试 file_utils 模块的核心功能
覆盖 find_json_files, parse_json_file, find_image_file, get_relative_path,
json_loads, json_dumps_str, write_json_file, ORJSON_AVAILABLE, create_file_link 等
"""

import json
import logging
import platform
import shutil
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from unsloth_finetune.data.labelme.file_utils import (
    ORJSON_AVAILABLE,
    create_file_link,
    find_image_file,
    find_json_files,
    get_relative_path,
    json_dumps_str,
    json_loads,
    parse_json_file,
    write_json_file,
)
from unsloth_finetune.data.labelme.progress_logger import (
    SUPPORTED_IMAGE_EXTENSIONS,
    create_progress_bar,
    setup_progress_logging,
    TQDM_AVAILABLE,
)


# ============================================================
# find_json_files 测试
# ============================================================

class TestFindJsonFiles:

    @pytest.fixture
    def json_dir(self, tmp_path):
        sub_dir = tmp_path / "subdir"
        sub_dir.mkdir()
        (tmp_path / "a.json").write_text("{}", encoding="utf-8")
        (tmp_path / "b.json").write_text("{}", encoding="utf-8")
        (tmp_path / "c.txt").write_text("text", encoding="utf-8")
        (sub_dir / "d.json").write_text("{}", encoding="utf-8")
        return tmp_path

    def test_recursive_find(self, json_dir):
        result = find_json_files(json_dir, recursive=True)
        names = [p.name for p in result]
        assert len(result) == 3
        assert "a.json" in names
        assert "b.json" in names
        assert "d.json" in names

    def test_non_recursive_find(self, json_dir):
        result = find_json_files(json_dir, recursive=False)
        names = [p.name for p in result]
        assert len(result) == 2
        assert "a.json" in names
        assert "b.json" in names

    def test_nonexistent_dir(self):
        result = find_json_files(Path("/nonexistent"))
        assert result == []

    def test_sorted_result(self, json_dir):
        result = find_json_files(json_dir, recursive=True)
        for i in range(len(result) - 1):
            assert str(result[i]) <= str(result[i + 1])

    def test_with_logger(self, json_dir):
        logger = setup_progress_logging("TestFindJson", use_tqdm=False)
        result = find_json_files(json_dir, recursive=True, logger=logger)
        assert len(result) == 3

    def test_empty_dir(self, tmp_path):
        result = find_json_files(tmp_path, recursive=True)
        assert result == []


# ============================================================
# parse_json_file 测试
# ============================================================

class TestParseJsonFile:

    @pytest.fixture
    def parse_dir(self, tmp_path):
        logger = setup_progress_logging("TestParseJson", use_tqdm=False)

        valid_json = tmp_path / "valid.json"
        valid_json.write_text('{"key": "value"}', encoding="utf-8")

        utf8_json = tmp_path / "utf8.json"
        data = {"label": "中文标签"}
        utf8_json.write_text(json.dumps(data), encoding="utf-8")

        gbk_json = tmp_path / "gbk.json"
        data = {"label": "中文标签"}
        gbk_json.write_text(json.dumps(data), encoding="gbk")

        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("{bad json content}", encoding="utf-8")

        return tmp_path, logger

    def test_valid_utf8(self, parse_dir):
        test_dir, logger = parse_dir
        result = parse_json_file(test_dir / "valid.json", logger)
        assert result is not None
        assert result["key"] == "value"

    def test_utf8_chinese(self, parse_dir):
        test_dir, logger = parse_dir
        result = parse_json_file(test_dir / "utf8.json", logger)
        assert result is not None
        assert result["label"] == "中文标签"

    def test_gbk_fallback(self, parse_dir):
        test_dir, logger = parse_dir
        result = parse_json_file(test_dir / "gbk.json", logger)
        assert result is not None
        assert result["label"] == "中文标签"

    def test_invalid_json(self, parse_dir):
        test_dir, logger = parse_dir
        result = parse_json_file(test_dir / "invalid.json", logger)
        assert result is None

    def test_nonexistent_file(self, parse_dir):
        test_dir, logger = parse_dir
        # parse_json_file 对不存在文件会抛出 FileNotFoundError
        with pytest.raises(FileNotFoundError):
            parse_json_file(Path("/nonexistent.json"), logger)

    def test_without_logger(self, parse_dir):
        test_dir, _ = parse_dir
        result = parse_json_file(test_dir / "valid.json")
        assert result is not None
        assert result["key"] == "value"

    def test_nested_json(self, tmp_path):
        nested_file = tmp_path / "nested.json"
        nested_file.write_text(
            '{"shapes": [{"label": "cat"}, {"label": "dog"}]}',
            encoding="utf-8",
        )
        result = parse_json_file(nested_file)
        assert result is not None
        assert len(result["shapes"]) == 2


# ============================================================
# find_image_file 测试
# ============================================================

class TestFindImageFile:

    @pytest.fixture
    def image_dir(self, tmp_path):
        json_path = tmp_path / "test_image.json"
        data = {"imagePath": "test_image.jpg"}
        json_path.write_text(json.dumps(data), encoding="utf-8")

        image_path = tmp_path / "test_image.jpg"
        image_path.write_bytes(b"\xff\xd8\xff\xe0")

        other_image = tmp_path / "other_name.jpg"
        other_image.write_bytes(b"\xff\xd8\xff\xe0")

        return tmp_path

    def test_no_image_path_str(self, image_dir):
        json_path = image_dir / "test_image.json"
        result = find_image_file(json_path)
        assert result is not None
        assert result.stem == "test_image"

    def test_strict_name_match_valid(self, image_dir):
        json_path = image_dir / "test_image.json"
        result = find_image_file(json_path, "test_image.jpg", strict_name_match=True)
        assert result is not None
        assert result.name == "test_image.jpg"

    def test_strict_name_match_invalid_path_but_found_by_stem(self, image_dir):
        json_path = image_dir / "test_image.json"
        result = find_image_file(json_path, "other_name.jpg", strict_name_match=True)
        assert result is not None
        assert result.stem == "test_image"

    def test_strict_name_match_no_match_at_all(self, image_dir):
        json_no_match = image_dir / "unique_file.json"
        json_no_match.write_text(
            '{"imagePath": "other_name.jpg"}', encoding="utf-8"
        )
        result = find_image_file(json_no_match, "other_name.jpg", strict_name_match=True)
        assert result is None

    def test_loose_name_match(self, image_dir):
        json_path = image_dir / "test_image.json"
        result = find_image_file(json_path, "test_image.jpg", strict_name_match=False)
        assert result is not None

    def test_loose_name_match_fallback_to_stem(self, image_dir):
        json_only_stem = image_dir / "test_image.json"
        json_only_stem.write_text(
            '{"imagePath": "nonexistent.jpg"}', encoding="utf-8"
        )
        result = find_image_file(
            json_only_stem, "nonexistent.jpg", strict_name_match=False
        )
        assert result is not None
        assert result.stem == "test_image"

    def test_no_image_path_str_no_match(self, image_dir):
        json_no_image = image_dir / "orphan.json"
        json_no_image.write_text('{"shapes": []}', encoding="utf-8")
        result = find_image_file(json_no_image)
        assert result is None

    def test_png_extension(self, image_dir):
        json_path = image_dir / "photo.json"
        json_path.write_text('{"imagePath": "photo.jpg"}', encoding="utf-8")
        png_path = image_dir / "photo.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        result = find_image_file(json_path)
        assert result is not None
        assert result.suffix in [".png", ".jpg"]

    def test_custom_supported_extensions(self, image_dir):
        json_path = image_dir / "test_image.json"
        result = find_image_file(
            json_path, supported_extensions={".png", ".webp"}
        )
        assert result is None


# ============================================================
# get_relative_path 测试
# ============================================================

class TestGetRelativePath:

    def test_relative_within_base(self):
        result = get_relative_path(
            Path("/home/user/data/file.json"), Path("/home/user/data")
        )
        assert result == Path("file.json")

    def test_relative_with_subdir(self):
        result = get_relative_path(
            Path("/home/user/data/sub/file.json"), Path("/home/user/data")
        )
        assert result == Path("sub/file.json")

    def test_outside_base(self):
        result = get_relative_path(
            Path("/other/path/file.json"), Path("/home/user/data")
        )
        assert result == Path("file.json")


# ============================================================
# json_loads 测试
# ============================================================

class TestJsonLoads:

    def test_basic_string(self):
        data = '{"key": "value", "num": 42}'
        result = json_loads(data)
        assert result["key"] == "value"
        assert result["num"] == 42

    def test_bytes_input(self):
        data = b'{"key": "value"}'
        result = json_loads(data)
        assert result["key"] == "value"

    def test_nested_structure(self):
        data = '{"shapes": [{"label": "cat"}, {"label": "dog"}]}'
        result = json_loads(data)
        assert len(result["shapes"]) == 2

    def test_chinese_content(self):
        data = '{"label": "中文标签"}'
        result = json_loads(data)
        assert result["label"] == "中文标签"

    def test_empty_dict(self):
        result = json_loads("{}")
        assert result == {}

    def test_list_input(self):
        data = '[1, 2, 3]'
        result = json_loads(data)
        assert result == [1, 2, 3]


# ============================================================
# json_dumps_str 测试
# ============================================================

class TestJsonDumpsStr:

    def test_basic_object(self):
        obj = {"key": "value"}
        result = json_dumps_str(obj)
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_with_indent(self):
        obj = {"key": "value", "nested": {"a": 1}}
        result = json_dumps_str(obj, indent=2)
        assert "\n" in result
        parsed = json.loads(result)
        assert parsed["nested"]["a"] == 1

    def test_no_indent(self):
        obj = {"key": "value"}
        result = json_dumps_str(obj, indent=None)
        assert "\n" not in result

    def test_chinese_content(self):
        obj = {"label": "中文标签"}
        result = json_dumps_str(obj)
        assert "中文标签" in result

    def test_ensure_ascii_false(self):
        obj = {"label": "中文"}
        result = json_dumps_str(obj, ensure_ascii=False)
        assert "中文" in result

    def test_list_object(self):
        obj = [1, 2, 3]
        result = json_dumps_str(obj)
        parsed = json.loads(result)
        assert parsed == [1, 2, 3]


# ============================================================
# write_json_file 测试
# ============================================================

class TestWriteJsonFile:

    def test_basic_write(self, tmp_path):
        file_path = tmp_path / "output.json"
        data = {"key": "value", "num": 42}
        write_json_file(file_path, data)
        assert file_path.exists()
        content = json.loads(file_path.read_text(encoding="utf-8"))
        assert content["key"] == "value"
        assert content["num"] == 42

    def test_write_with_indent(self, tmp_path):
        file_path = tmp_path / "formatted.json"
        data = {"key": "value"}
        write_json_file(file_path, data, indent=2)
        content = file_path.read_text(encoding="utf-8")
        assert "\n" in content

    def test_write_nested_structure(self, tmp_path):
        file_path = tmp_path / "nested.json"
        data = {"shapes": [{"label": "cat"}, {"label": "dog"}]}
        write_json_file(file_path, data)
        content = json.loads(file_path.read_text(encoding="utf-8"))
        assert len(content["shapes"]) == 2

    def test_write_chinese_content(self, tmp_path):
        file_path = tmp_path / "chinese.json"
        data = {"label": "中文标签"}
        write_json_file(file_path, data)
        content = json.loads(file_path.read_text(encoding="utf-8"))
        assert content["label"] == "中文标签"

    def test_auto_create_parent_dirs(self, tmp_path):
        file_path = tmp_path / "subdir" / "deep" / "output.json"
        data = {"key": "value"}
        write_json_file(file_path, data)
        assert file_path.exists()

    def test_write_empty_dict(self, tmp_path):
        file_path = tmp_path / "empty.json"
        write_json_file(file_path, {})
        content = json.loads(file_path.read_text(encoding="utf-8"))
        assert content == {}


# ============================================================
# ORJSON_AVAILABLE 测试
# ============================================================

class TestOrjsonAvailable:

    def test_is_boolean(self):
        assert isinstance(ORJSON_AVAILABLE, bool)

    def test_json_loads_works_with_orjson_or_stdlib(self):
        # Regardless of whether orjson is available, json_loads should work
        data = '{"key": "value"}'
        result = json_loads(data)
        assert result["key"] == "value"

    def test_json_dumps_str_works_with_orjson_or_stdlib(self):
        obj = {"key": "value"}
        result = json_dumps_str(obj)
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_write_json_file_works_with_orjson_or_stdlib(self, tmp_path):
        file_path = tmp_path / "test_orjson.json"
        data = {"test": True}
        write_json_file(file_path, data)
        content = json.loads(file_path.read_text(encoding="utf-8"))
        assert content["test"] is True


# ============================================================
# create_file_link 测试
# ============================================================

class TestCreateFileLink:

    @pytest.fixture
    def link_dir(self, tmp_path):
        source_file = tmp_path / "source.txt"
        source_file.write_text("hello world", encoding="utf-8")
        target_path = tmp_path / "target.txt"
        return source_file, target_path

    def test_copy_mode(self, link_dir):
        source, target = link_dir
        success, method = create_file_link(source, target, link_type="copy")
        assert success is True
        assert "复制" in method or "copy" in method.lower()
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "hello world"

    def test_source_not_exists(self, tmp_path):
        source = tmp_path / "nonexistent.txt"
        target = tmp_path / "target.txt"
        success, method = create_file_link(source, target, link_type="copy")
        assert success is False

    def test_source_is_directory(self, tmp_path):
        source_dir = tmp_path / "dir"
        source_dir.mkdir()
        target = tmp_path / "target.txt"
        success, method = create_file_link(source_dir, target, link_type="copy")
        assert success is False

    def test_auto_mode(self, link_dir):
        source, target = link_dir
        success, method = create_file_link(source, target, link_type="auto")
        assert success is True
        assert target.exists()

    def test_unknown_link_type(self, link_dir):
        source, target = link_dir
        success, method = create_file_link(source, target, link_type="unknown_type")
        assert success is False

    def test_target_exists_overwrite(self, link_dir):
        source, target = link_dir
        target.write_text("old content", encoding="utf-8")
        success, method = create_file_link(source, target, link_type="copy")
        assert success is True
        assert target.read_text(encoding="utf-8") == "hello world"

    def test_hardlink_mode(self, link_dir):
        source, target = link_dir
        success, method = create_file_link(source, target, link_type="hardlink")
        # On Windows, hardlink may succeed or fail depending on permissions
        # We just check the function doesn't crash
        if success:
            assert target.exists()

    def test_symlink_mode(self, link_dir):
        source, target = link_dir
        success, method = create_file_link(source, target, link_type="symlink")
        # On Windows, symlink may fail without admin rights
        if success:
            assert target.exists()

    def test_auto_creates_parent_dir(self, tmp_path):
        source = tmp_path / "source.txt"
        source.write_text("content", encoding="utf-8")
        target = tmp_path / "subdir" / "deep" / "target.txt"
        success, method = create_file_link(source, target, link_type="copy")
        assert success is True
        assert target.parent.exists()


# ============================================================
# 多线程一致性测试 (保留原有测试)
# ============================================================

class TestMultithreadingConsistency:

    @pytest.fixture
    def multi_json_dir(self, tmp_path):
        num_files = 50
        for i in range(num_files):
            json_path = tmp_path / f"file_{i:03d}.json"
            data = {
                "imagePath": f"file_{i:03d}.jpg",
                "shapes": [
                    {"label": f"label_{i % 5}", "shape_type": "rectangle"}
                ],
            }
            json_path.write_text(json.dumps(data), encoding="utf-8")
        return tmp_path

    def test_find_json_files_consistency(self, multi_json_dir):
        result = find_json_files(multi_json_dir, recursive=True)
        assert len(result) == 50

    def test_parse_json_files_parallel(self, multi_json_dir):
        from concurrent.futures import ThreadPoolExecutor

        json_files = find_json_files(multi_json_dir, recursive=True)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(parse_json_file, json_files))
        assert len(results) == 50
        non_none = [r for r in results if r is not None]
        assert len(non_none) == 50
