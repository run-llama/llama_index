"""Test img_utils."""

import base64
from io import BytesIO

import pytest
from PIL import Image

from llama_index.core.img_utils import b64_2_img, img_2_b64


def test_img_2_b64_returns_string() -> None:
    """Test that img_2_b64 returns a string, not bytes."""
    # Create a simple test image
    img = Image.new("RGB", (10, 10), color="red")

    # Convert to base64
    b64_str = img_2_b64(img)

    # Assert it's a string
    assert isinstance(b64_str, str)

    # Verify it's valid base64 by decoding it back
    decoded_bytes = base64.b64decode(b64_str)
    assert isinstance(decoded_bytes, bytes)


def test_b64_2_img_round_trip() -> None:
    """Test that b64_2_img can decode what img_2_b64 produces."""
    # Create a simple test image
    original_img = Image.new("RGB", (10, 10), color="blue")

    # Convert to base64 and back
    b64_str = img_2_b64(original_img)
    reconstructed_img = b64_2_img(b64_str)

    # Check that the images are the same size
    assert reconstructed_img.size == original_img.size

    # Check that the mode is the same
    assert reconstructed_img.mode == original_img.mode