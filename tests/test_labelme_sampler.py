"""
测试 labelme_sampler 模块的核心功能
覆盖 SelectionMode, ImageLabelInfo, SelectionResult, BalancedSelectionResult, LabelMeSampler
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from labelme_tools.labelme_sampler import (
    BalancedSelectionResult,
    ImageLabelInfo,
    LabelMeSampler,
    SelectionMode,
    SelectionResult,
    select_balanced_samples,
)


# ============================================================
# SelectionMode 测试
# ============================================================

class TestSelectionMode:

    def test_mode_values(self):
        assert SelectionMode.N_IMAGES.value == "n_images"
        assert SelectionMode.N_LABELS.value == "n_labels"

    def test_enum_members(self):
        members = list(SelectionMode)
        assert len(members) == 2

    def test_is_enum(self):
        from enum import Enum
        assert issubclass(SelectionMode, Enum)


# ============================================================
# ImageLabelInfo 测试
# ============================================================

class TestImageLabelInfo:

    def test_basic_creation(self):
        info = ImageLabelInfo(
            json_path="/path/to/file.json",
            image_path="/path/to/image.jpg",
            label_counts={"cat": 3, "dog": 2},
        )
        assert info.json_path == "/path/to/file.json"
        assert info.image_path == "/path/to/image.jpg"
        assert info.label_counts["cat"] == 3
        assert info.total_labels == 5

    def test_post_init_total_labels(self):
        info = ImageLabelInfo(
            json_path="/path/to/file.json",
            label_counts={"cat": 2, "dog": 1},
        )
        assert info.total_labels == 3

    def test_empty_label_counts(self):
        info = ImageLabelInfo(
            json_path="/path/to/file.json",
            label_counts={},
        )
        assert info.total_labels == 0

    def test_none_image_path(self):
        info = ImageLabelInfo(
            json_path="/path/to/file.json",
            label_counts={"cat": 1},
        )
        assert info.image_path is None


# ============================================================
# SelectionResult 测试
# ============================================================

class TestSelectionResult:

    def test_default_values(self):
        result = SelectionResult(
            category="cat",
            mode=SelectionMode.N_IMAGES,
            target_count=10,
        )
        assert result.category == "cat"
        assert result.mode == SelectionMode.N_IMAGES
        assert result.target_count == 10
        assert result.selected_images == []
        assert result.total_selected_images == 0
        assert result.total_selected_labels == 0
        assert result.has_duplicates is False
        assert result.duplicate_count == 0

    def test_duration_property(self):
        result = SelectionResult(
            category="cat",
            mode=SelectionMode.N_IMAGES,
            target_count=10,
        )
        result.start_time = datetime(2024, 1, 1)
        result.end_time = datetime(2024, 1, 1, 0, 0, 5)
        assert result.duration == 5.0

    def test_duration_none(self):
        result = SelectionResult(
            category="cat",
            mode=SelectionMode.N_IMAGES,
            target_count=10,
        )
        assert result.duration is None

    def test_to_dict(self):
        info = ImageLabelInfo(
            json_path="/path/a.json",
            label_counts={"cat": 2},
        )
        result = SelectionResult(
            category="cat",
            mode=SelectionMode.N_IMAGES,
            target_count=10,
            selected_images=[info],
            total_selected_images=1,
            total_selected_labels=2,
        )
        d = result.to_dict()
        assert d["category"] == "cat"
        assert d["mode"] == "n_images"
        assert d["target_count"] == 10
        assert d["total_selected_images"] == 1
        assert d["total_selected_labels"] == 2
        assert len(d["selected_images"]) == 1


# ============================================================
# BalancedSelectionResult 测试
# ============================================================

class TestBalancedSelectionResult:

    def test_default_values(self):
        result = BalancedSelectionResult(
            source_dir="/path/to/source",
            mode=SelectionMode.N_IMAGES,
            target_count=100,
        )
        assert result.source_dir == "/path/to/source"
        assert result.mode == SelectionMode.N_IMAGES
        assert result.target_count == 100
        assert result.random_seed is None
        assert result.category_results == {}
        assert result.total_selected_images == 0
        assert result.unique_images == set()

    def test_duration_property(self):
        result = BalancedSelectionResult(
            source_dir="/path/to/source",
            mode=SelectionMode.N_IMAGES,
            target_count=100,
        )
        result.start_time = datetime(2024, 1, 1)
        result.end_time = datetime(2024, 1, 1, 0, 1, 0)
        assert result.duration == 60.0

    def test_unique_image_count(self):
        result = BalancedSelectionResult(
            source_dir="/path/to/source",
            mode=SelectionMode.N_IMAGES,
            target_count=100,
            unique_images={"/path/a.json", "/path/b.json", "/path/c.json"},
        )
        assert result.unique_image_count == 3

    def test_to_dict(self):
        result = BalancedSelectionResult(
            source_dir="/path/to/source",
            mode=SelectionMode.N_IMAGES,
            target_count=100,
            total_selected_images=5,
            unique_images={"/a.json", "/b.json"},
        )
        d = result.to_dict()
        assert d["source_dir"] == "/path/to/source"
        assert d["mode"] == "n_images"
        assert d["target_count"] == 100
        assert d["total_selected_images"] == 5
        assert d["unique_image_count"] == 2


# ============================================================
# LabelMeSampler 内部方法测试
# ============================================================

class TestLabelMeSamplerInternal:

    @pytest.fixture
    def sampler(self, tmp_path):
        return LabelMeSampler(
            source_dir=str(tmp_path),
            use_tqdm=False,
            validate_images=False,
        )

    def test_extract_label_counts(self, sampler):
        data = {
            "shapes": [
                {"label": "cat"},
                {"label": "cat"},
                {"label": "dog"},
            ]
        }
        result = sampler._extract_label_counts(data)
        assert result["cat"] == 2
        assert result["dog"] == 1

    def test_extract_label_counts_empty_shapes(self, sampler):
        data = {"shapes": []}
        result = sampler._extract_label_counts(data)
        assert result == {}

    def test_extract_label_counts_no_shapes_key(self, sampler):
        data = {}
        result = sampler._extract_label_counts(data)
        assert result == {}

    def test_extract_label_counts_invalid_shapes(self, sampler):
        data = {"shapes": "invalid"}
        result = sampler._extract_label_counts(data)
        assert result == {}

    def test_extract_label_counts_empty_label(self, sampler):
        data = {"shapes": [{"label": ""}, {"label": "cat"}]}
        result = sampler._extract_label_counts(data)
        assert "" not in result
        assert "cat" in result

    def test_select_n_images_enough_images(self, sampler):
        images = [
            ImageLabelInfo(json_path=f"/path/{i}.json", label_counts={"cat": i + 1})
            for i in range(10)
        ]
        result = sampler._select_n_images(images, 5)
        assert result.total_selected_images == 5
        assert result.has_duplicates is False

    def test_select_n_images_insufficient_images(self, sampler):
        images = [
            ImageLabelInfo(json_path=f"/path/{i}.json", label_counts={"cat": 1})
            for i in range(3)
        ]
        result = sampler._select_n_images(images, 10)
        assert result.total_selected_images == 10
        assert result.has_duplicates is True
        assert result.duplicate_count > 0

    def test_select_n_labels_enough_labels(self, sampler):
        images = [
            ImageLabelInfo(json_path=f"/path/{i}.json", label_counts={"cat": 5})
            for i in range(10)
        ]
        result = sampler._select_n_labels(images, 20, "cat")
        assert result.total_selected_labels >= 20

    def test_select_n_labels_insufficient_labels(self, sampler):
        images = [
            ImageLabelInfo(json_path=f"/path/{i}.json", label_counts={"cat": 2})
            for i in range(3)
        ]
        result = sampler._select_n_labels(images, 10, "cat")
        assert result.has_duplicates is True
        assert result.total_selected_labels >= 10


# ============================================================
# LabelMeSampler 流程测试
# ============================================================

class TestLabelMeSamplerFlow:

    @pytest.fixture
    def sampler_dir(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        for i in range(10):
            json_path = source_dir / f"img_{i:03d}.json"
            label = "cat" if i < 5 else "dog"
            count = i % 3 + 1
            shapes = [{"label": label} for _ in range(count)]
            data = {
                "imagePath": f"img_{i:03d}.jpg",
                "shapes": shapes,
            }
            json_path.write_text(json.dumps(data), encoding="utf-8")
            (source_dir / f"img_{i:03d}.jpg").write_bytes(
                b"\xff\xd8\xff\xe0" + b"\x00" * 200
            )

        return source_dir

    def test_select_samples_n_images(self, sampler_dir):
        sampler = LabelMeSampler(
            source_dir=str(sampler_dir),
            mode=SelectionMode.N_IMAGES,
            target_count=3,
            use_tqdm=False,
            validate_images=True,
        )
        result = sampler.select_samples()
        assert result.mode == SelectionMode.N_IMAGES
        assert len(result.category_results) > 0

    def test_select_samples_n_labels(self, sampler_dir):
        sampler = LabelMeSampler(
            source_dir=str(sampler_dir),
            mode=SelectionMode.N_LABELS,
            target_count=5,
            use_tqdm=False,
            validate_images=True,
        )
        result = sampler.select_samples()
        assert result.mode == SelectionMode.N_LABELS
        assert len(result.category_results) > 0

    def test_select_samples_no_validate_images(self, sampler_dir):
        sampler = LabelMeSampler(
            source_dir=str(sampler_dir),
            mode=SelectionMode.N_IMAGES,
            target_count=5,
            use_tqdm=False,
            validate_images=False,
        )
        result = sampler.select_samples()
        assert len(result.category_results) > 0

    def test_select_samples_no_json_files(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        sampler = LabelMeSampler(
            source_dir=str(empty_dir),
            use_tqdm=False,
        )
        result = sampler.select_samples()
        assert len(result.category_results) == 0
        assert result.total_selected_images == 0

    def test_get_selected_files(self, sampler_dir):
        sampler = LabelMeSampler(
            source_dir=str(sampler_dir),
            mode=SelectionMode.N_IMAGES,
            target_count=3,
            use_tqdm=False,
            validate_images=False,
        )
        files = sampler.get_selected_files()
        assert isinstance(files, list)
        assert len(files) > 0

    def test_random_seed_reproducibility(self, sampler_dir):
        sampler1 = LabelMeSampler(
            source_dir=str(sampler_dir),
            mode=SelectionMode.N_IMAGES,
            target_count=3,
            random_seed=42,
            use_tqdm=False,
            validate_images=False,
        )
        result1 = sampler1.select_samples()

        sampler2 = LabelMeSampler(
            source_dir=str(sampler_dir),
            mode=SelectionMode.N_IMAGES,
            target_count=3,
            random_seed=42,
            use_tqdm=False,
            validate_images=False,
        )
        result2 = sampler2.select_samples()

        # With same seed, total selected should be the same
        assert result1.total_selected_images == result2.total_selected_images


# ============================================================
# select_balanced_samples 便捷函数测试
# ============================================================

class TestSelectBalancedSamples:

    def test_function_returns_result(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        json_path = source_dir / "img.json"
        json_path.write_text(
            json.dumps({
                "imagePath": "img.jpg",
                "shapes": [{"label": "cat"}],
            }),
            encoding="utf-8",
        )
        (source_dir / "img.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 200)

        result = select_balanced_samples(
            source_dir=str(source_dir),
            mode="n_images",
            target_count=1,
            use_tqdm=False,
        )
        assert isinstance(result, BalancedSelectionResult)
        assert result.mode == SelectionMode.N_IMAGES

    def test_mode_n_labels(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        json_path = source_dir / "img.json"
        json_path.write_text(
            json.dumps({
                "imagePath": "img.jpg",
                "shapes": [{"label": "cat"}, {"label": "cat"}],
            }),
            encoding="utf-8",
        )
        (source_dir / "img.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 200)

        result = select_balanced_samples(
            source_dir=str(source_dir),
            mode="n_labels",
            target_count=2,
            use_tqdm=False,
        )
        assert isinstance(result, BalancedSelectionResult)
        assert result.mode == SelectionMode.N_LABELS