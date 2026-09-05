"""
core.image_loader
------------------
Handles reading an image from disk, validating that it is a real, decodable
image PixelPulse can work with, and exposing basic metadata used throughout
the rest of the pipeline.
"""

import os
import cv2

from config import settings


class ImageLoadError(Exception):
    """Raised when an image cannot be located, read, or decoded."""


class LoadedImage:
    """
    Thin wrapper around a decoded image plus its metadata.

    Attributes
    ----------
    path : str
        Original filesystem path.
    bgr : np.ndarray
        Image in OpenCV's native BGR color order, at original resolution.
    gray : np.ndarray
        Single-channel grayscale version of the image.
    height, width : int
        Original image dimensions in pixels.
    """

    def __init__(self, path, bgr, gray):
        self.path = path
        self.bgr = bgr
        self.gray = gray
        self.height, self.width = gray.shape[:2]

    @property
    def resolution_str(self):
        return f"{self.width} x {self.height}"


def validate_path(path):
    """Raise ImageLoadError with a human-readable reason if `path` is unusable."""
    if not os.path.exists(path):
        raise ImageLoadError(f"File not found: {path}")
    if not os.path.isfile(path):
        raise ImageLoadError(f"Not a file: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in settings.SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {sorted(settings.SUPPORTED_EXTENSIONS)}"
        )

    if os.path.getsize(path) == 0:
        raise ImageLoadError(f"File is empty: {path}")


def load_image(path):
    """
    Load and validate an image from disk.

    Returns
    -------
    LoadedImage

    Raises
    ------
    ImageLoadError
        If the path is invalid, unreadable, or not decodable as an image.
    """
    validate_path(path)

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ImageLoadError(
            f"OpenCV could not decode '{path}'. The file may be corrupted "
            f"or is not a valid image despite its extension."
        )

    if bgr.shape[0] < 16 or bgr.shape[1] < 16:
        raise ImageLoadError(
            f"Image too small to analyze meaningfully ({bgr.shape[1]}x{bgr.shape[0]}). "
            f"Minimum supported dimension is 16px."
        )

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return LoadedImage(path, bgr, gray)
