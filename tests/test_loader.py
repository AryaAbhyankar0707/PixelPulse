"""
Tests for core.image_loader.

Uses synthetic images written to a pytest tmp_path so tests never depend
on the sample_images/ directory contents.
"""

import numpy as np
import cv2
import pytest

from core.image_loader import load_image, ImageLoadError


def _write_image(path, height=64, width=64):
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def test_load_valid_image(tmp_path):
    img_path = _write_image(tmp_path / "valid.png")
    loaded = load_image(str(img_path))

    assert loaded.width == 64
    assert loaded.height == 64
    assert loaded.bgr.shape == (64, 64, 3)
    assert loaded.gray.shape == (64, 64)
    assert loaded.resolution_str == "64 x 64"


def test_missing_file_raises(tmp_path):
    with pytest.raises(ImageLoadError):
        load_image(str(tmp_path / "does_not_exist.jpg"))


def test_unsupported_extension_raises(tmp_path):
    bad_path = tmp_path / "notes.txt"
    bad_path.write_text("this is not an image")
    with pytest.raises(ImageLoadError):
        load_image(str(bad_path))


def test_empty_file_raises(tmp_path):
    empty_path = tmp_path / "empty.jpg"
    empty_path.touch()
    with pytest.raises(ImageLoadError):
        load_image(str(empty_path))


def test_too_small_image_raises(tmp_path):
    img_path = _write_image(tmp_path / "tiny.png", height=8, width=8)
    with pytest.raises(ImageLoadError):
        load_image(str(img_path))


def test_corrupt_image_raises(tmp_path):
    fake_path = tmp_path / "fake.jpg"
    fake_path.write_bytes(b"not really jpeg data" * 10)
    with pytest.raises(ImageLoadError):
        load_image(str(fake_path))
