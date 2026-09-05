"""
enhancement.sharpen
----------------------
Unsharp masking: blur the image, subtract the blurred version from the
original to isolate high-frequency detail, then add that detail back in
with a gain factor. This is the standard, controllable way to sharpen
without the harsh artifacts of a naive Laplacian-kernel sharpen.
"""

import cv2
import numpy as np


def apply(bgr, amount=1.0, radius=2):
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=radius)
    sharpened = cv2.addWeighted(bgr, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
