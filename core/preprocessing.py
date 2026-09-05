"""
core.preprocessing
-------------------
Prepares a loaded image for analysis. Kept deliberately separate from
image_loader so that "reading a file" and "getting it analysis-ready" are
independently testable steps, matching the pipeline diagram in the report.
"""

import cv2

from config import settings


def resize_for_analysis(gray, bgr, max_dim=settings.MAX_ANALYSIS_DIMENSION):
    """
    Downscale both the grayscale and BGR arrays so the longer side is at
    most `max_dim`, preserving aspect ratio. Large images are downscaled
    purely for analysis speed and consistency across the metrics (several
    thresholds in settings.py were calibrated at this working resolution);
    the original-resolution arrays are kept separately for anything that
    should operate at full size (e.g. saving an enhanced output).

    Returns
    -------
    (gray_resized, bgr_resized, scale_factor)
    """
    h, w = gray.shape[:2]
    longer_side = max(h, w)

    if longer_side <= max_dim:
        return gray, bgr, 1.0

    scale = max_dim / float(longer_side)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    gray_resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    bgr_resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return gray_resized, bgr_resized, scale


def denoise_light(gray):
    """A very light Gaussian pass used only to stabilize measurements that
    are otherwise overly sensitive to single-pixel sensor noise (e.g. edge
    density). Not part of the enhancement pipeline."""
    return cv2.GaussianBlur(gray, (3, 3), 0)
