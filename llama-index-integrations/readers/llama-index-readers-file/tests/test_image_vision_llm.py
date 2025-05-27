from __future__ import annotations

from contextlib import contextmanager

import builtins

import pytest
from unittest import mock

from typing import Dict, List
from types import ModuleType

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
    """
    This double fakes the `Blip2Processor` tokenizer object so as to
    avoid having to instantiate the actual tokenizer for these tests.
    """

    def __call__(self, img, prompt, return_tensors) -> TokenizerFake:
        """
        This is just a stub for the purposes of the test,
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
    """
    This double fakes the `Blip2ForConditionalGeneration` model object
    in order to avoid having to download checkpoints for these tests.
    """

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
        """
        This is just a dummy method for the purposes of the test (it
        needs to be defined, but is not used). Hence, we return nothing.
        """


@contextmanager
def _get_custom_import(torch_installed: bool):
    """
    Simulate absence of PyTorch installation depending on the input flag.

    Args:
        torch_installed (bool): Flag indicating whether or not PyTorch is installed.

    Returns:
        Generator: Parametrized `_custom_import()` function.

    """
    # Store the original __import__ function
    original_import = builtins.__import__

    def _custom_import(module_name: str, *args, **kwargs) -> ModuleType:
        """
        If `torch_installed` is False, act as if PyTorch is not installed.
        """
        if module_name == "torch" and not torch_installed:
            raise ImportError('No module named "torch.')

        return original_import(module_name, *args, **kwargs)

    try:
        # Replace the built-in __import__ function
        builtins.__import__ = _custom_import

        yield
    except Exception:
        # Restore the original import function
        builtins.__import__ = original_import

        raise
    finally:
        # Restore the original import function
        builtins.__import__ = original_import


@pytest.mark.skipif(
    Image is None,
    reason="PIL not installed",
)
@pytest.mark.parametrize(
    "torch_installed",
    [
        pytest.param(
            False,
            id="torch_not_installed",
        ),
        pytest.param(
            True,
            id="torch_installed",
        ),
    ],
)
def test_image_vision_llm_reader_load_data_with_parser_config(
    torch_installed: bool, test_16x16_png_image_file: str
):
    """
    We use doubles (mocks and fakes) for the model and the tokenizer objects
    in order to avoid having to download checkpoints as part of tests, while
    still covering all essential `ImageVisionLLMReader` class functionality.
    """
    with (
        mock.patch(
            "transformers.Blip2ForConditionalGeneration.from_pretrained",
            return_value=ModelFake(),
        ) as model,
        mock.patch(
            "transformers.Blip2Processor.from_pretrained",
            return_value=TokenizerFake(),
        ) as processor,
    ):
        parser_config = {
            "processor": processor(),
            "model": model(),
            "device": "auto",  # not used (placeholder)
            "dtype": float,  # not used (placeholder)
        }

        if torch_installed:
            image_vision_llm_reader = ImageVisionLLMReader(
                parser_config=parser_config, keep_image=True
            )
            assert image_vision_llm_reader._torch_imported
        else:
            with _get_custom_import(torch_installed=False):
                image_vision_llm_reader = ImageVisionLLMReader(
                    parser_config=parser_config, keep_image=True
                )
                assert not image_vision_llm_reader._torch_imported

        result = image_vision_llm_reader.load_data(file=test_16x16_png_image_file)[0]
        assert (
            result.text
            == "Question: describe what you see in this image. Answer: a black and white checkered pattern"
        )


@pytest.mark.skipif(
    Image is None,
    reason="PIL not installed",
)
@pytest.mark.parametrize(
    "torch_installed",
    [
        pytest.param(
            False,
            id="torch_not_installed",
        ),
        pytest.param(
            True,
            id="torch_installed",
        ),
    ],
)
def test_image_vision_llm_reader_load_data_wo_parser_config(
    torch_installed: bool, test_16x16_png_image_file: str
):
    """
    We use doubles (mocks and fakes) for the model and the tokenizer objects
    in order to avoid having to download checkpoints as part of tests, while
    still covering most of the `ImageVisionLLMReader` class functionality.
    """
    with (
        mock.patch(
            "transformers.Blip2ForConditionalGeneration.from_pretrained",
            return_value=ModelFake(),
        ),
        mock.patch(
            "transformers.Blip2Processor.from_pretrained",
            return_value=TokenizerFake(),
        ),
    ):
        if torch_installed:
            image_vision_llm_reader = ImageVisionLLMReader()
            result = image_vision_llm_reader.load_data(file=test_16x16_png_image_file)[
                0
            ]
            assert (
                result.text
                == "Question: describe what you see in this image. Answer: a black and white checkered pattern"
            )
        else:
            with _get_custom_import(torch_installed=False):
                with pytest.raises(ImportError) as excinfo:
                    image_vision_llm_reader = ImageVisionLLMReader()

                assert (
                    str(excinfo.value)
                    == "Please install extra dependencies that are required for the ImageCaptionReader: `pip install torch transformers sentencepiece Pillow`"
                )
