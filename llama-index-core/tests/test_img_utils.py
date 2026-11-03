import json

from PIL import Image

from llama_index.core.img_utils import b64_2_img, img_2_b64


def test_img_2_b64_returns_str() -> None:
    """img_2_b64 is annotated `-> str` and must actually return str."""
    image = Image.new("RGB", (2, 2), color=(255, 0, 0))

    b64 = img_2_b64(image)

    assert isinstance(b64, str)


def test_img_2_b64_round_trip_and_text_usage() -> None:
    """The returned value must be usable as text and round-trip via b64_2_img."""
    image = Image.new("RGB", (2, 2), color=(0, 255, 0))

    b64 = img_2_b64(image)

    json.dumps({"image": b64})
    assert f"data:image/jpeg;base64,{b64}".startswith("data:image/jpeg;base64,")
    assert b64_2_img(b64).size == (2, 2)
