"""
analysis.histogram_analysis
----------------------------
Brightness and contrast, derived from the grayscale intensity histogram.

Concepts demonstrated: histogram computation, mean/standard-deviation as
distributional statistics, and threshold-based categorical labeling.
"""

import cv2
import numpy as np

from config import settings


def compute_histogram(gray):
    """Return the 256-bin grayscale histogram, normalized to sum to 1."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    return hist / total if total > 0 else hist


def analyze(gray):
    """
    Returns
    -------
    dict with:
        brightness_mean : float (0-255)
        brightness_label: str
        contrast_std    : float
        contrast_label  : str
        histogram       : np.ndarray (256,) normalized histogram, for plotting
    """
    hist = compute_histogram(gray)
    mean = float(np.mean(gray))
    std = float(np.std(gray))

    if mean <= settings.BRIGHTNESS_DARK_MAX:
        brightness_label = "DARK"
    elif mean <= settings.BRIGHTNESS_LOW_MAX:
        brightness_label = "LOW"
    elif mean >= settings.BRIGHTNESS_OVEREXPOSED_MIN:
        brightness_label = "OVEREXPOSED"
    elif mean >= settings.BRIGHTNESS_HIGH_MIN:
        brightness_label = "HIGH"
    else:
        brightness_label = "NORMAL"

    if std <= settings.CONTRAST_LOW_MAX:
        contrast_label = "LOW"
    elif std >= settings.CONTRAST_GOOD_MIN:
        contrast_label = "GOOD"
    else:
        contrast_label = "MODERATE"

    # Normalize contrast std onto a 0-100 scale for the score engine.
    # A std of ~64 (quarter of the full 0-255 range) is treated as the
    # practical ceiling for "excellent" contrast in natural photographs.
    contrast_score = float(np.clip((std / 64.0) * 100, 0, 100))

    # Brightness score peaks at mid-range and falls off toward both
    # extremes (very dark or very overexposed images lose points).
    ideal = 127.5
    distance = abs(mean - ideal) / ideal  # 0 at ideal, 1 at pure black/white
    brightness_score = float(np.clip((1 - distance) * 100, 0, 100))

    return {
        "brightness_mean": round(mean, 2),
        "brightness_label": brightness_label,
        "brightness_score": round(brightness_score, 2),
        "contrast_std": round(std, 2),
        "contrast_label": contrast_label,
        "contrast_score": round(contrast_score, 2),
        "histogram": hist,
    }
