"""
analysis.blur_analysis
------------------------
Sharpness / blur detection via variance of the Laplacian: a sharp image has
lots of high-frequency content, so convolving with the Laplacian kernel and
looking at the variance of the response gives a simple, well-known blur
metric. Low variance -> flat, blurred edges. High variance -> crisp edges.

Concepts demonstrated: Laplacian (second-derivative) edge operator,
convolution, variance as a focus/sharpness measure.
"""

import cv2
import numpy as np

from config import settings


def laplacian_variance(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def analyze(gray):
    variance = laplacian_variance(gray)

    if variance < settings.BLUR_VARIANCE_LOW:
        label = "SEVERE"
    elif variance < settings.BLUR_VARIANCE_MODERATE:
        label = "MODERATE"
    elif variance < settings.BLUR_VARIANCE_HIGH:
        label = "MILD"
    else:
        label = "SHARP"

    # Score: 0 at variance=0, saturating to 100 at BLUR_VARIANCE_HIGH
    score = float(np.clip((variance / settings.BLUR_VARIANCE_HIGH) * 100, 0, 100))

    return {
        "laplacian_variance": round(variance, 2),
        "blur_label": label,
        "sharpness_score": round(score, 2),
    }
