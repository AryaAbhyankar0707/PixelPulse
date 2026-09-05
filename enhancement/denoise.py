"""
enhancement.denoise
---------------------
Applies Non-Local Means denoising -- chosen over a plain Gaussian/median
blur because it averages similar patches from across the whole image
rather than just a local neighborhood, so it removes noise while
preserving edges far better than simple smoothing.
"""

import cv2


def apply(bgr, strength=7):
    """strength: filter strength `h` for luminance channel (higher = more smoothing)."""
    return cv2.fastNlMeansDenoisingColored(bgr, None, h=strength, hColor=strength,
                                            templateWindowSize=7, searchWindowSize=21)
