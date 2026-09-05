"""
analysis.noise_analysis
-------------------------
Estimates sensor/compression noise using a high-frequency-residual method:
subtract a heavily median-filtered version of the image from the original;
what remains is dominated by noise rather than genuine structure, so its
standard deviation is a reasonable proxy for noise level.

Concepts demonstrated: median filtering, residual analysis, standard
deviation as a noise metric (a lightweight relative of the classic
wavelet-subband noise estimators).
"""

import cv2
import numpy as np

from config import settings


def estimate_noise_sigma(gray):
    """Return an estimated noise standard deviation (0 = perfectly clean)."""
    denoised = cv2.medianBlur(gray, 5)
    residual = gray.astype(np.float32) - denoised.astype(np.float32)
    # Robust std estimate (median absolute deviation scaled) is less
    # sensitive to a few strong edges leaking into the residual than a
    # plain standard deviation would be.
    mad = np.median(np.abs(residual - np.median(residual)))
    sigma = 1.4826 * mad
    return float(sigma)


def analyze(gray):
    sigma = estimate_noise_sigma(gray)

    if sigma <= settings.NOISE_SIGMA_LOW:
        label = "LOW"
    elif sigma <= settings.NOISE_SIGMA_MODERATE:
        label = "MODERATE"
    elif sigma <= settings.NOISE_SIGMA_HIGH:
        label = "HIGH"
    else:
        label = "SEVERE"

    # Score: 100 at sigma=0, decaying to 0 by NOISE_SIGMA_HIGH*1.5
    ceiling = settings.NOISE_SIGMA_HIGH * 1.5
    score = float(np.clip((1 - sigma / ceiling) * 100, 0, 100))

    return {
        "noise_sigma": round(sigma, 3),
        "noise_label": label,
        "noise_score": round(score, 2),
    }
