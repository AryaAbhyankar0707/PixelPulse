"""
analysis.edge_analysis
----------------------
Structural-detail analysis using Canny edges plus gradient strength.

Canny provides a sparse structural map, while robust gradient statistics
provide a fallback when a low-contrast image contains real structure that
Canny's hysteresis thresholds do not capture. Combining both signals makes
edge scoring less brittle across photographs with different contrast levels.
"""

import cv2
import numpy as np

from config import settings


def compute_edge_map(gray):
    """Return a Canny edge map using robust percentile thresholds."""
    gray = np.asarray(gray, dtype=np.uint8)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    median = float(np.median(smooth))
    lower = int(np.clip(0.66 * median, 10, 180))
    upper = int(np.clip(1.33 * median, lower + 10, 240))
    return cv2.Canny(smooth, lower, upper)


def _gradient_strength(gray):
    """Robust normalized gradient strength, capped to avoid outlier dominance."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    # The 75th percentile captures broad structural detail without letting a
    # few extreme boundaries dominate the score.
    p75 = float(np.percentile(magnitude, 75))
    return p75


def analyze(gray):
    edges = compute_edge_map(gray)
    edge_density = float(np.count_nonzero(edges)) / edges.size
    gradient_strength = _gradient_strength(gray)

    density_score = np.clip(
        (edge_density / settings.EDGE_DENSITY_GOOD_MIN) * 85, 0, 100
    )
    gradient_score = np.clip(
        (gradient_strength / settings.GRADIENT_STRENGTH_GOOD) * 100, 0, 100
    )

    if edge_density <= settings.EDGE_DENSITY_LOW_MAX:
        label = "WEAK" if gradient_strength < settings.GRADIENT_STRENGTH_LOW else "MODERATE"
    elif edge_density >= settings.EDGE_DENSITY_GOOD_MIN:
        label = "GOOD"
    else:
        label = "MODERATE"

    # When Canny produces no edges, do not fabricate edge density from a
    # fallback detector. Gradient strength still provides a useful structural
    # signal, but it receives a conservative contribution to the final score.
    if edge_density <= settings.EDGE_DENSITY_LOW_MAX:
        score = float(np.clip(0.40 * gradient_score, 0, 100))
    else:
        score = float(np.clip(0.70 * density_score + 0.30 * gradient_score, 0, 100))

    return {
        "edge_density": round(edge_density, 5),
        "gradient_strength": round(gradient_strength, 2),
        "edge_label": label,
        "edge_score": round(score, 2),
        "edge_map": edges,
    }
