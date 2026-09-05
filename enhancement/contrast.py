"""
enhancement.contrast
-----------------------
CLAHE (Contrast Limited Adaptive Histogram Equalization) applied on the L
channel of LAB color space -- this improves local contrast and corrects
under/overexposure while avoiding both the color shifts of equalizing RGB
channels independently and the over-amplification of noise that plain
(non-adaptive, non-clipped) histogram equalization causes.
"""

import cv2


def apply(bgr, clip_limit=2.5, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
