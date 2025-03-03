from __future__ import annotations

import pytest
from unittest import mock

from typing import Dict, List

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


class TokenizerFake:
    def __call__(self, img, prompt, return_tensors) -> TokenizerFake:
        """This is just a stub for the purposes of the test,
        so we just return the instance itself.
        """
        return self

    def to(self, device, dtype) -> Dict[str, list]:
        """
        The output is the tokenized version of the prompt
        "Question: describe what you see in this image. Answer:"
        It should be of type `transformers.image_processing_base.BatchFeature`
        with `torch.Tensor` typed values for `"input_ids"`, `"attention_mask"`,
        and `"pixel_values"` keys. However, we will fake them as lists of
        integers where values are needed (`None` elsewhere) in order
        to not require `torch` or `numpy` imports.
        """
        return {
            "input_ids": [
                [2, 45641, 35, 6190, 99, 47, 192, 11, 42, 2274, 4, 31652, 35]
            ],
            "attention_mask": [[None]],
            "pixel_values": [[[[None]]]],
        }

    def decode(
        self, tokens: Dict[str, List[int]], skip_special_tokens: bool = True
    ) -> str:
        """
        We return the known expected decoded response for the
        `test_16x16_png_image_file` fixture and the default prompt
        of the `ImageVisionLLMReader` class.
        """
        return "Question: describe what you see in this image. Answer: a black and white checkered pattern"


class ModelFake:
    def generate(self, **kwargs) -> list:
        """
        The output is the tokenized version of the prompt
        "Question: describe what you see in this image. \
            Answer: a black and white checkered pattern"
        It should be of type `torch.Tensor`. However, we will fake it as a
        list of integers order to not require `torch` or `numpy` imports.
        """
        return [
            [
                2,
                45641,
                35,
                6190,
                99,
                47,
                192,
                11,
                42,
                2274,
                4,
                31652,
                35,
                10,
                909,
                8,
                1104,
                5851,
                438,
                20093,
                6184,
                50118,
            ]
        ]

    def to(self, device) -> None:
        """This is just a dummy method for the purposes of the test (it
        needs to be defined, but is not used). Hence, we return nothing.
        """


@pytest.mark.skipif(
    Image is None,
    reason="PIL not installed",
)
def test_image_vision_llm_reader(test_16x16_png_image_file: str):
    """
    We use doubles (mocks and fakes) for the model and the tokenizer objects
    in order to avoid having to download checkpoints as part of tests, while
    still covering most of the `ImageVisionLLMReader` class functionality.
    """
    with mock.patch(
        "transformers.Blip2ForConditionalGeneration.from_pretrained",
        return_value=ModelFake(),
    ), mock.patch(
        "transformers.Blip2Processor.from_pretrained",
        return_value=TokenizerFake(),
    ):
        image_vision_llm_reader = ImageVisionLLMReader()
        result = image_vision_llm_reader.load_data(file=test_16x16_png_image_file)[0]
        assert (
            result.text
            == "Question: describe what you see in this image. Answer: a black and white checkered pattern"
        )
