"""
Tests for the analysis/ modules.

Each test builds a small synthetic grayscale image with a known property
(sharp vs. blurred, flat vs. high-contrast, noisy vs. clean) and checks
that the corresponding metric moves in the correct direction. These are
relative/directional checks rather than checks against magic numbers,
since the exact thresholds in config/settings.py are tunable.
"""

import numpy as np
import cv2

from analysis import (
    histogram_analysis,
    noise_analysis,
    blur_analysis,
    edge_analysis,
    texture_analysis,
)


def _checkerboard(size=128, square=8):
    """A crisp, high-contrast, high-texture synthetic test pattern."""
    tile = np.indices((size, size)).sum(axis=0) // square % 2
    return (tile * 255).astype(np.uint8)


def _flat_gray(size=128, value=128):
    """A perfectly flat image: no edges, no texture, no contrast."""
    return np.full((size, size), value, dtype=np.uint8)


def test_blur_analysis_detects_sharpening_difference():
    sharp = _checkerboard()
    blurred = cv2.GaussianBlur(sharp, (15, 15), 0)

    sharp_result = blur_analysis.analyze(sharp)
    blurred_result = blur_analysis.analyze(blurred)

    assert sharp_result["laplacian_variance"] > blurred_result["laplacian_variance"]
    assert sharp_result["sharpness_score"] > blurred_result["sharpness_score"]


def test_histogram_analysis_contrast_ordering():
    flat = _flat_gray()
    checkered = _checkerboard()

    flat_result = histogram_analysis.analyze(flat)
    checkered_result = histogram_analysis.analyze(checkered)

    assert flat_result["contrast_std"] < checkered_result["contrast_std"]
    assert flat_result["contrast_score"] < checkered_result["contrast_score"]
    assert flat_result["contrast_label"] == "LOW"


def test_histogram_analysis_brightness_extremes():
    dark = _flat_gray(value=10)
    bright = _flat_gray(value=250)
    mid = _flat_gray(value=128)

    dark_result = histogram_analysis.analyze(dark)
    bright_result = histogram_analysis.analyze(bright)
    mid_result = histogram_analysis.analyze(mid)

    # Mid-gray should score at least as well on brightness as either extreme.
    assert mid_result["brightness_score"] >= dark_result["brightness_score"]
    assert mid_result["brightness_score"] >= bright_result["brightness_score"]


def test_noise_analysis_detects_added_noise():
    rng = np.random.default_rng(42)
    clean = _flat_gray()
    noisy = np.clip(
        clean.astype(np.int16) + rng.normal(0, 25, clean.shape), 0, 255
    ).astype(np.uint8)

    clean_result = noise_analysis.analyze(clean)
    noisy_result = noise_analysis.analyze(noisy)

    assert noisy_result["noise_sigma"] > clean_result["noise_sigma"]
    assert noisy_result["noise_score"] < clean_result["noise_score"]


def test_edge_analysis_flat_vs_structured():
    flat = _flat_gray()
    checkered = _checkerboard()

    flat_result = edge_analysis.analyze(flat)
    checkered_result = edge_analysis.analyze(checkered)

    assert flat_result["edge_density"] < checkered_result["edge_density"]
    assert flat_result["edge_score"] < checkered_result["edge_score"]
    assert flat_result["edge_map"].shape == flat.shape


def test_texture_analysis_flat_vs_structured():
    flat = _flat_gray()
    checkered = _checkerboard()

    flat_result = texture_analysis.analyze(flat)
    checkered_result = texture_analysis.analyze(checkered)

    assert flat_result["texture_energy"] < checkered_result["texture_energy"]
    assert flat_result["texture_score"] < checkered_result["texture_score"]


def test_edge_analysis_uses_gradient_fallback_for_low_contrast_structure():
    x = np.linspace(60, 190, 128, dtype=np.float32)
    gradient = np.tile(x, (128, 1)).astype(np.uint8)
    result = edge_analysis.analyze(gradient)
    assert result["gradient_strength"] > 0
    assert result["edge_score"] > 0
