"""Image preprocessing: bytes -> normalized float32 numpy array.

Deliberately PyTorch-independent. Produces a (1, 3, 64, 64) float32 array
that the classifier converts to a torch.Tensor at the classifier boundary.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from galaxeye.config import IMAGE_SIZE, MEAN, STD


class ImageDecodeError(ValueError):
    """Raised when input bytes cannot be decoded as a valid image."""


def decode_image(data: bytes) -> Image.Image:
    """Decode raw bytes into an RGB PIL image.

    Raises ImageDecodeError for undecodable or unsupported input.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # PIL raises OSError / UnidentifiedImageError etc.
        raise ImageDecodeError(f"Could not decode image: {exc}") from exc
    return image.convert("RGB")


def to_input_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a normalized (1, 3, 64, 64) float32 array.

    Converts to RGB, resizes to IMAGE_SIZE, scales pixels to [0, 1], and
    normalizes per channel with ImageNet mean/std (required by the frozen
    ResNet backbone).
    """
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0  # (64, 64, 3) in [0, 1]
    array = array.transpose(2, 0, 1)                     # (3, 64, 64)
    mean = np.asarray(MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(STD, dtype=np.float32).reshape(3, 1, 1)
    array = (array - mean) / std                         # normalize per channel
    return array[np.newaxis, ...]                        # (1, 3, 64, 64)
