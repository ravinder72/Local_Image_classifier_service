"""Tests for image decoding and preprocessing."""

import io

import numpy as np
import pytest
from PIL import Image

from galaxeye.config import IMAGE_SIZE, MEAN, STD
from galaxeye.preprocess import ImageDecodeError, decode_image, to_input_array


def _bytes(mode="RGB", size=(64, 64), color=(128, 128, 128), fmt="PNG"):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format=fmt)
    return buf.getvalue()


def test_decode_valid_png_returns_rgb():
    image = decode_image(_bytes())
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (64, 64)


def test_decode_valid_jpeg_returns_rgb():
    image = decode_image(_bytes(fmt="JPEG"))
    assert image.mode == "RGB"


def test_decode_converts_grayscale_to_rgb():
    image = decode_image(_bytes(mode="L", color=100))
    assert image.mode == "RGB"


def test_decode_invalid_bytes_raises():
    with pytest.raises(ImageDecodeError):
        decode_image(b"this is not an image")


def test_decode_empty_bytes_raises():
    with pytest.raises(ImageDecodeError):
        decode_image(b"")


def test_to_input_array_shape_and_dtype():
    arr = to_input_array(Image.new("RGB", (64, 64), (100, 150, 200)))
    assert arr.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert arr.dtype == np.float32


def test_to_input_array_resizes_non_64_image():
    arr = to_input_array(Image.new("RGB", (128, 32), (10, 20, 30)))
    assert arr.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_to_input_array_black_image_normalized():
    arr = to_input_array(Image.new("RGB", (64, 64), (0, 0, 0)))
    expected = np.array(
        [(0.0 - m) / s for m, s in zip(MEAN, STD)], dtype=np.float32
    )
    np.testing.assert_allclose(arr[0, :, 0, 0], expected, rtol=1e-5)


def test_to_input_array_white_image_normalized():
    arr = to_input_array(Image.new("RGB", (64, 64), (255, 255, 255)))
    expected = np.array(
        [(1.0 - m) / s for m, s in zip(MEAN, STD)], dtype=np.float32
    )
    np.testing.assert_allclose(arr[0, :, 0, 0], expected, rtol=1e-5)


def test_to_input_array_preserves_channel_order():
    arr = to_input_array(Image.new("RGB", (64, 64), (255, 0, 0)))
    r = (1.0 - MEAN[0]) / STD[0]
    g = (0.0 - MEAN[1]) / STD[1]
    b = (0.0 - MEAN[2]) / STD[2]
    np.testing.assert_allclose(arr[0, :, 0, 0], [r, g, b], rtol=1e-5)


def test_decode_and_preprocess_end_to_end():
    arr = to_input_array(decode_image(_bytes()))
    assert arr.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert np.isfinite(arr).all()
