"""OpenAI Image Generation tool sppec.."""

import base64
import os
import time
from typing import Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_CACHE_DIR = "../../../img_cache"
DEFAULT_SIZE = "1024x1024"  # Dall-e-3 only supports 1024x1024


class OpenAIImageGenerationToolSpec(BaseToolSpec):
    """OpenAI Image Generation tool spec."""

    spec_functions = ["image_generation"]

    def __init__(self, api_key: str, cache_dir: Optional[str] = None) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "Please install openai with `pip install openai` to use this tool"
            )

        """Initialize with parameters."""
        self.client = OpenAI(api_key=api_key)
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR

    def get_cache_dir(self):
        return self.cache_dir

    def save_base64_image(self, base64_str, image_name):
        try:
            from io import BytesIO

            from PIL import Image
        except ImportError:
            raise ImportError(
                "Please install Pillow with `pip install Pillow` to use this tool"
            )
        cache_dir = self.cache_dir

        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Decode the base64 string
        image_data = base64.b64decode(base64_str)

        # Create an image from the decoded bytes and save it
        image_path = os.path.join(cache_dir, image_name)
        with Image.open(BytesIO(image_data)) as img:
            img.save(image_path)

        return image_path

    def image_generation(
        self,
        text: str,
        model: Optional[str] = "dall-e-3",
        quality: Optional[str] = "standard",
        num_images: Optional[int] = 1,
    ) -> str:
        """
        This tool accepts a natural language string and will use OpenAI's DALL-E model to generate an image.

        Args:
            text (str): The text to generate an image from.
            size (str): The size of the image to generate (1024x1024, 256x256, 512x512).
            model (str): The model to use to generate the image (dall-e-3, dall-e-2).
            quality (str): The quality of the image to generate (standard, hd).
            num_images (int): The number of images to generate.
        """
        response = self.client.images.generate(
            model=model,
            prompt=text,
            size=DEFAULT_SIZE,
            quality=quality,
            n=num_images,
            response_format="b64_json",
        )

        image_bytes = response.data[0].b64_json

        filename = f"{time.time()}.jpg"

        return self.save_base64_image(image_bytes, filename)
