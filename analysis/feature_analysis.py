"""
analysis.feature_analysis
----------------------------
Structural/feature-level analysis: HOG (Histogram of Oriented Gradients)
descriptor statistics and SIFT keypoint stability. These are diagnostic
rather than score-affecting -- they describe *what kind* of visual content
is present and how reliably it could be matched/tracked, which is useful
context in the final report even though it does not feed the 0-100 quality
score directly (documented explicitly to avoid double-counting sharpness
and edge signal that HOG/SIFT partially re-derive).

Concepts demonstrated: gradient-orientation histograms (HOG), scale-space
keypoint detection and descriptor extraction (SIFT).
"""

import cv2
import numpy as np
from skimage.feature import hog


def analyze(gray):
    # --- HOG -------------------------------------------------------------
    # Resize to a fixed working size so HOG cell/block geometry is stable
    # across differently-sized input images.
    working = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    hog_vector = hog(
        working,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )
    hog_energy = float(np.mean(hog_vector ** 2))

    # --- SIFT --------------------------------------------------------------
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    keypoint_count = len(keypoints)
    avg_response = (
        float(np.mean([kp.response for kp in keypoints])) if keypoint_count else 0.0
    )

    if keypoint_count >= 300:
        stability_label = "HIGH"
    elif keypoint_count >= 80:
        stability_label = "MODERATE"
    else:
        stability_label = "LOW"

    return {
        "hog_vector_length": int(hog_vector.shape[0]),
        "hog_energy": round(hog_energy, 5),
        "sift_keypoint_count": keypoint_count,
        "sift_avg_response": round(avg_response, 6),
        "feature_stability_label": stability_label,
    }
