"""Tests for llama_index.core.img_utils."""

import base64
import json
from io import BytesIO

from llama_index.core.img_utils import b64_2_img, img_2_b64
from PIL import Image
from PIL.ImageFile import ImageFile


def _image() -> ImageFile:
    buff = BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buff, format="JPEG")
    buff.seek(0)
    return Image.open(buff)


def test_img_2_b64_returns_str_not_bytes() -> None:
    """
    Regression test for #21186.

    img_2_b64 is annotated `-> str` but returned the bytes from base64.b64encode()
    wrapped in typing.cast, which only satisfies the type checker and does nothing at
    runtime. Callers (the file readers) annotate the result `image_str: Optional[str]`
    and put it on a Document, so the bytes surfaced later as
    "TypeError: Object of type bytes is not JSON serializable", or as a literal b'...'
    interpolated into a data URI.
    """
    encoded = img_2_b64(_image())

    assert isinstance(encoded, str)
    # the bytes repr must not leak through string interpolation
    assert not f"{encoded}".startswith("b'")
    # and the value must survive JSON serialization, as Document storage requires
    assert json.loads(json.dumps({"image": encoded}))["image"] == encoded


def test_img_2_b64_round_trips_through_b64_2_img() -> None:
    """The str returned by img_2_b64 must still decode back into an image."""
    image = _image()

    restored = b64_2_img(img_2_b64(image))

    assert restored.size == image.size


def test_img_2_b64_matches_plain_b64encode() -> None:
    """The encoding itself is unchanged; only its type is."""
    image = _image()
    buff = BytesIO()
    image.save(buff, format="JPEG")

    assert img_2_b64(image) == base64.b64encode(buff.getvalue()).decode("utf-8")
