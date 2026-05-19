"""
测试 color_utils 模块的核心功能
覆盖 hex_to_rgb, rgb_to_hex, srgb_to_linear, get_relative_luminance,
calculate_contrast_ratio, wcag_compliance
"""

import pytest

from unsloth_finetune.tools.color_contrast.color_utils import (
    calculate_contrast_ratio,
    get_relative_luminance,
    hex_to_rgb,
    rgb_to_hex,
    srgb_to_linear,
    wcag_compliance,
)


# ============================================================
# hex_to_rgb 测试
# ============================================================

class TestHexToRgb:

    def test_full_hex_with_hash(self):
        assert hex_to_rgb("#ff0000") == (255, 0, 0)

    def test_full_hex_without_hash(self):
        assert hex_to_rgb("ff0000") == (255, 0, 0)

    def test_short_hex_with_hash(self):
        assert hex_to_rgb("#f00") == (255, 0, 0)

    def test_short_hex_without_hash(self):
        assert hex_to_rgb("f00") == (255, 0, 0)

    def test_white(self):
        assert hex_to_rgb("#ffffff") == (255, 255, 255)

    def test_black(self):
        assert hex_to_rgb("#000000") == (0, 0, 0)

    def test_green(self):
        assert hex_to_rgb("#00ff00") == (0, 255, 0)

    def test_blue(self):
        assert hex_to_rgb("#0000ff") == (0, 0, 255)

    def test_mid_gray(self):
        assert hex_to_rgb("#808080") == (128, 128, 128)

    def test_short_hex_expansion(self):
        # #abc → a=aa, b=bb, c=cc
        result = hex_to_rgb("#abc")
        assert result == (170, 187, 204)

    def test_hex_with_whitespace(self):
        assert hex_to_rgb("  #ff0000  ") == (255, 0, 0)

    def test_uppercase_hex(self):
        assert hex_to_rgb("#FF0000") == (255, 0, 0)

    def test_mixed_case_hex(self):
        assert hex_to_rgb("#Ff00Aa") == (255, 0, 170)

    def test_all_channels_same(self):
        assert hex_to_rgb("#333333") == (51, 51, 51)

    def test_short_hex_all_same(self):
        assert hex_to_rgb("#333") == (51, 51, 51)


# ============================================================
# rgb_to_hex 测试
# ============================================================

class TestRgbToHex:

    def test_red(self):
        assert rgb_to_hex(255, 0, 0) == "#ff0000"

    def test_green(self):
        assert rgb_to_hex(0, 255, 0) == "#00ff00"

    def test_blue(self):
        assert rgb_to_hex(0, 0, 255) == "#0000ff"

    def test_white(self):
        assert rgb_to_hex(255, 255, 255) == "#ffffff"

    def test_black(self):
        assert rgb_to_hex(0, 0, 0) == "#000000"

    def test_mid_gray(self):
        assert rgb_to_hex(128, 128, 128) == "#808080"

    def test_low_values(self):
        assert rgb_to_hex(1, 2, 3) == "#010203"

    def test_roundtrip_with_hex_to_rgb(self):
        original = "#336699"
        rgb = hex_to_rgb(original)
        result = rgb_to_hex(*rgb)
        assert result == original

    def test_roundtrip_various_colors(self):
        colors = ["#ff0000", "#00ff00", "#0000ff", "#ffffff", "#000000", "#808080"]
        for color in colors:
            rgb = hex_to_rgb(color)
            result = rgb_to_hex(*rgb)
            assert result == color


# ============================================================
# srgb_to_linear 测试
# ============================================================

class TestSrgbToLinear:

    def test_zero(self):
        assert srgb_to_linear(0) == 0.0

    def test_255(self):
        result = srgb_to_linear(255)
        assert abs(result - 1.0) < 1e-10

    def test_low_value_linear_region(self):
        # Channel 10 → c = 10/255 ≈ 0.03922, which is ≤ 0.03928
        result = srgb_to_linear(10)
        expected = (10 / 255.0) / 12.92
        assert abs(result - expected) < 1e-10

    def test_high_value_power_region(self):
        # Channel 128 → c = 128/255 ≈ 0.502, which is > 0.03928
        result = srgb_to_linear(128)
        c = 128 / 255.0
        expected = ((c + 0.055) / 1.055) ** 2.4
        assert abs(result - expected) < 1e-10

    def test_threshold_boundary(self):
        # The threshold is at c = 0.03928, which corresponds to channel ≈ 10.04
        # Channel 10 should use linear formula
        result_10 = srgb_to_linear(10)
        # Channel 11 should use power formula
        result_11 = srgb_to_linear(11)
        # Both should be close since they're near the boundary
        assert result_10 < result_11

    def test_monotonically_increasing(self):
        values = [srgb_to_linear(i) for i in range(0, 256)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]

    def test_50_percent_gray(self):
        # sRGB 128 is approximately 21.4% linear
        result = srgb_to_linear(128)
        assert 0.2 < result < 0.22


# ============================================================
# get_relative_luminance 测试
# ============================================================

class TestGetRelativeLuminance:

    def test_black(self):
        result = get_relative_luminance(0, 0, 0)
        assert abs(result - 0.0) < 1e-10

    def test_white(self):
        result = get_relative_luminance(255, 255, 255)
        assert abs(result - 1.0) < 1e-10

    def test_red(self):
        result = get_relative_luminance(255, 0, 0)
        # Red contributes: 0.2126 * 1.0 = 0.2126
        expected = 0.2126 * srgb_to_linear(255)
        assert abs(result - expected) < 1e-6

    def test_green(self):
        result = get_relative_luminance(0, 255, 0)
        # Green contributes: 0.7152 * 1.0 = 0.7152
        expected = 0.7152 * srgb_to_linear(255)
        assert abs(result - expected) < 1e-6

    def test_blue(self):
        result = get_relative_luminance(0, 0, 255)
        # Blue contributes: 0.0722 * 1.0 = 0.0722
        expected = 0.0722 * srgb_to_linear(255)
        assert abs(result - expected) < 1e-6

    def test_mid_gray(self):
        result = get_relative_luminance(128, 128, 128)
        # All channels same, so luminance = srgb_to_linear(128)
        expected = srgb_to_linear(128)
        assert abs(result - expected) < 1e-6

    def test_green_brighter_than_red(self):
        green_lum = get_relative_luminance(0, 255, 0)
        red_lum = get_relative_luminance(255, 0, 0)
        assert green_lum > red_lum

    def test_green_brighter_than_blue(self):
        green_lum = get_relative_luminance(0, 255, 0)
        blue_lum = get_relative_luminance(0, 0, 255)
        assert green_lum > blue_lum

    def test_values_between_0_and_1(self):
        for r, g, b in [(10, 20, 30), (100, 150, 200), (200, 200, 200)]:
            result = get_relative_luminance(r, g, b)
            assert 0.0 <= result <= 1.0


# ============================================================
# calculate_contrast_ratio 测试
# ============================================================

class TestCalculateContrastRatio:

    def test_black_white(self):
        ratio = calculate_contrast_ratio("#ffffff", "#000000")
        assert abs(ratio - 21.0) < 0.01

    def test_white_black_order(self):
        ratio = calculate_contrast_ratio("#000000", "#ffffff")
        assert abs(ratio - 21.0) < 0.01

    def test_same_color(self):
        ratio = calculate_contrast_ratio("#ff0000", "#ff0000")
        assert abs(ratio - 1.0) < 0.01

    def test_two_grays(self):
        ratio = calculate_contrast_ratio("#808080", "#404040")
        assert ratio > 1.0
        assert ratio < 21.0

    def test_red_green(self):
        ratio = calculate_contrast_ratio("#ff0000", "#00ff00")
        assert abs(ratio - 2.91) < 0.1

    def test_contrast_ratio_range(self):
        # Contrast ratio should always be between 1 and 21
        colors = [
            ("#ffffff", "#000000"),
            ("#ff0000", "#00ff00"),
            ("#333333", "#333333"),
            ("#abcdef", "#123456"),
        ]
        for c1, c2 in colors:
            ratio = calculate_contrast_ratio(c1, c2)
            assert 1.0 <= ratio <= 21.0

    def test_symmetry(self):
        ratio1 = calculate_contrast_ratio("#ffffff", "#000000")
        ratio2 = calculate_contrast_ratio("#000000", "#ffffff")
        assert abs(ratio1 - ratio2) < 0.01

    def test_without_hash_prefix(self):
        ratio1 = calculate_contrast_ratio("#ffffff", "#000000")
        ratio2 = calculate_contrast_ratio("ffffff", "000000")
        assert abs(ratio1 - ratio2) < 0.01

    def test_short_hex_format(self):
        ratio = calculate_contrast_ratio("#fff", "#000")
        assert abs(ratio - 21.0) < 0.01


# ============================================================
# wcag_compliance 测试
# ============================================================

class TestWcagCompliance:

    def test_high_contrast_passes_all(self):
        result = wcag_compliance(21.0)
        assert result["AA_normal"] is True
        assert result["AA_large"] is True
        assert result["AAA_normal"] is True
        assert result["AAA_large"] is True
        assert result["ratio"] == 21.0

    def test_aa_normal_threshold(self):
        result = wcag_compliance(4.5)
        assert result["AA_normal"] is True
        assert result["AA_large"] is True
        assert result["AAA_normal"] is False
        assert result["AAA_large"] is True

    def test_aa_large_threshold(self):
        result = wcag_compliance(3.0)
        assert result["AA_normal"] is False
        assert result["AA_large"] is True
        assert result["AAA_normal"] is False
        assert result["AAA_large"] is False

    def test_aaa_normal_threshold(self):
        result = wcag_compliance(7.0)
        assert result["AA_normal"] is True
        assert result["AA_large"] is True
        assert result["AAA_normal"] is True
        assert result["AAA_large"] is True

    def test_aaa_large_threshold(self):
        result = wcag_compliance(4.5)
        assert result["AAA_large"] is True

    def test_low_contrast_fails_all(self):
        result = wcag_compliance(1.5)
        assert result["AA_normal"] is False
        assert result["AA_large"] is False
        assert result["AAA_normal"] is False
        assert result["AAA_large"] is False

    def test_below_aa_normal(self):
        result = wcag_compliance(4.0)
        assert result["AA_normal"] is False
        assert result["AA_large"] is True

    def test_ratio_preserved(self):
        result = wcag_compliance(5.7)
        assert result["ratio"] == 5.7

    def test_returns_dict(self):
        result = wcag_compliance(10.0)
        assert isinstance(result, dict)
        assert "AA_normal" in result
        assert "AA_large" in result
        assert "AAA_normal" in result
        assert "AAA_large" in result
        assert "ratio" in result

    def test_boundary_values(self):
        # Test just below and above thresholds
        assert wcag_compliance(4.49)["AA_normal"] is False
        assert wcag_compliance(4.50)["AA_normal"] is True
        assert wcag_compliance(2.99)["AA_large"] is False
        assert wcag_compliance(3.00)["AA_large"] is True
        assert wcag_compliance(6.99)["AAA_normal"] is False
        assert wcag_compliance(7.00)["AAA_normal"] is True


# ============================================================
# 集成测试: 从颜色到合规性
# ============================================================

class TestColorToComplianceIntegration:

    def test_black_white_full_compliance(self):
        ratio = calculate_contrast_ratio("#ffffff", "#000000")
        result = wcag_compliance(ratio)
        assert result["AA_normal"] is True
        assert result["AAA_normal"] is True

    def test_light_gray_on_white_fails(self):
        ratio = calculate_contrast_ratio("#ffffff", "#cccccc")
        result = wcag_compliance(ratio)
        assert result["AA_normal"] is False

    def test_dark_gray_on_white_passes_aa(self):
        ratio = calculate_contrast_ratio("#ffffff", "#767676")
        # Ratio should be approximately 4.54:1
        result = wcag_compliance(ratio)
        assert result["AA_normal"] is True

    def test_red_on_white(self):
        ratio = calculate_contrast_ratio("#ffffff", "#ff0000")
        result = wcag_compliance(ratio)
        # Red on white has ratio ~3.98, fails AA normal
        assert result["AA_normal"] is False
        assert result["AA_large"] is True
