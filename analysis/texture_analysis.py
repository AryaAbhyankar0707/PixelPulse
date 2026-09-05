"""
analysis.texture_analysis
----------------------------
Texture quality via a local standard-deviation map: for each pixel,
compute the standard deviation of intensities in a small neighborhood.
Regions with fine texture (grass, fabric, skin) produce high local std;
flat/washed-out regions produce near-zero local std. The variance of this
map across the whole image summarizes how much genuine texture survived
capture/compression.

Concepts demonstrated: sliding-window local statistics, box filtering
(implemented via mean-of-squares minus square-of-mean for speed).
"""

import cv2
import numpy as np

from config import settings


def local_std_map(gray, kernel_size=9):
    """Compute a per-pixel local standard deviation map efficiently using
    the identity Var(X) = E[X^2] - E[X]^2, both estimated with box filters."""
    img = gray.astype(np.float32)
    mean = cv2.boxFilter(img, ddepth=-1, ksize=(kernel_size, kernel_size))
    mean_sq = cv2.boxFilter(img * img, ddepth=-1, ksize=(kernel_size, kernel_size))
    variance = np.clip(mean_sq - mean * mean, 0, None)
    return np.sqrt(variance)


def analyze(gray):
    std_map = local_std_map(gray)
    texture_energy = float(np.var(std_map))

    if texture_energy <= settings.TEXTURE_LOW_MAX:
        label = "FLAT"
    elif texture_energy >= settings.TEXTURE_GOOD_MIN:
        label = "RICH"
    else:
        label = "MODERATE"

    score = float(np.clip((texture_energy / settings.TEXTURE_GOOD_MIN) * 100, 0, 100))

    return {
        "texture_energy": round(texture_energy, 2),
        "texture_label": label,
        "texture_score": round(score, 2),
    }
