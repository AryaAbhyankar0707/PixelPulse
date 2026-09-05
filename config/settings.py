"""
Central configuration for PixelPulse.

Every "magic number" used anywhere in the pipeline lives here so that the
scoring behaviour can be explained, tuned, and cited in the project report
without hunting through the codebase.
"""

# ---------------------------------------------------------------------------
# Quality Score Engine weights (must sum to 1.0)
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "sharpness": 0.25,
    "contrast": 0.20,
    "noise": 0.15,
    "illumination": 0.15,
    "edge_preservation": 0.15,
    "texture": 0.10,
}

# ---------------------------------------------------------------------------
# Quality category bands (inclusive lower bound, exclusive upper bound)
# ---------------------------------------------------------------------------
QUALITY_BANDS = [
    (0, 31, "POOR"),
    (31, 51, "WEAK"),
    (51, 71, "ACCEPTABLE"),
    (71, 86, "GOOD"),
    (86, 101, "EXCELLENT"),
]

# ---------------------------------------------------------------------------
# Blur detection (variance of Laplacian). Below this raw variance, an image
# is considered meaningfully blurred. Calibrated empirically on a mixed set
# of sharp / motion-blurred / defocused sample images.
# ---------------------------------------------------------------------------
BLUR_VARIANCE_LOW = 60.0      # below this -> "SEVERE" blur
BLUR_VARIANCE_MODERATE = 150.0  # below this -> "MODERATE" blur
BLUR_VARIANCE_HIGH = 400.0    # above this -> considered fully sharp

# ---------------------------------------------------------------------------
# Noise estimation (high-frequency residual standard deviation)
# ---------------------------------------------------------------------------
NOISE_SIGMA_LOW = 2.5
NOISE_SIGMA_MODERATE = 6.0
NOISE_SIGMA_HIGH = 12.0

# ---------------------------------------------------------------------------
# Illumination / brightness (mean pixel intensity, 0-255)
# ---------------------------------------------------------------------------
BRIGHTNESS_DARK_MAX = 70
BRIGHTNESS_LOW_MAX = 100
BRIGHTNESS_HIGH_MIN = 180
BRIGHTNESS_OVEREXPOSED_MIN = 220

# ---------------------------------------------------------------------------
# Contrast (standard deviation of grayscale histogram, 0-255 scale)
# ---------------------------------------------------------------------------
CONTRAST_LOW_MAX = 30
CONTRAST_GOOD_MIN = 45

# ---------------------------------------------------------------------------
# Edge preservation (fraction of pixels marked as edges by Canny)
# ---------------------------------------------------------------------------
EDGE_DENSITY_LOW_MAX = 0.015
EDGE_DENSITY_GOOD_MIN = 0.04
GRADIENT_STRENGTH_LOW = 15.0
GRADIENT_STRENGTH_GOOD = 60.0

# ---------------------------------------------------------------------------
# Texture (variance of a local standard-deviation map, i.e. how much
# fine-grained structure survives across the image)
# ---------------------------------------------------------------------------
TEXTURE_LOW_MAX = 80
TEXTURE_GOOD_MIN = 200

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
MAX_ANALYSIS_DIMENSION = 1600  # images larger than this are downscaled for analysis speed
