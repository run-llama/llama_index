import pytest

try:
    import torch
    import numpy as np
    from PIL import Image
except ImportError:
    torch = None
    np = None
    Image = None

from llama_index.readers.file.image_vision_llm.base import ImageVisionLLMReader


# Fixture to create a temporary 16x16 pixel image file
@pytest.fixture()
def test_16x16_png_image_file(tmp_path) -> str:
    # Create a checkerboard pattern (alternating 0 and 255)
    img_array = np.zeros((16, 16), dtype=np.uint8)
    img_array[::2, ::2] = 255  # Set even rows and columns to white
    img_array[1::2, 1::2] = 255  # Set odd rows and columns to white

    # Convert numpy array to PIL Image
    img = Image.fromarray(img_array, mode="L")  # 'L' mode is for grayscale

    file_path = tmp_path / "test_image_16x16.png"
    img.save(file_path)

    return file_path


@pytest.mark.skipif(
    torch is None or np is None or Image is None,
    reason="torch, numpy, PIL not installed",
)
@pytest.mark.slow()
def test_image_vision_llm_reader(test_16x16_png_image_file: str):
    image_vision_llm_reader = ImageVisionLLMReader()
    result = image_vision_llm_reader.load_data(file=test_16x16_png_image_file)[0]
    assert (
        result.text
        == "Question: describe what you see in this image. Answer: a black and white checkered pattern"
    )
