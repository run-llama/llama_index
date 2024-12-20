"""OpenAI Image Generation tool spec."""

import base64
import os
import time
from typing import Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_CACHE_DIR = "../../../img_cache"
DEFAULT_SIZE = "1024x1024"

valid_sizes = {
    "dall-e-2": ["256x256", "512x512", "1024x1024"],
    "dall-e-3": ["1024x1024", "1792x1024", "1024x1792"],
}


def get_extension(content: str):
    map = {
        "/": "jpg",
        "i": "png",
        "R": "gif",
        "U": "webp",
    }
    return map.get(content[0], "jpg")


class OpenAIImageGenerationToolSpec(BaseToolSpec):
    """OpenAI Image Generation tool spec."""

    spec_functions = ["image_generation"]

    def __init__(
        self, api_key: Optional[str] = None, cache_dir: Optional[str] = None
    ) -> None:
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
        size: Optional[str] = DEFAULT_SIZE,
        style: Optional[str] = "vivid",
        timeout: Optional[int] = None,
        download: Optional[bool] = False,
    ) -> str:
        """
        This tool accepts a natural language string and will use OpenAI's DALL-E model to generate an image.

        Args:
            text: The text to generate an image from.

            model: The model to use for image generation. Defaults to `dall-e-3`.
                Must be one of `dall-e-2` or `dall-e-3`.

            num_images: The number of images to generate. Defaults to 1.
                Must be between 1 and 10. For `dall-e-3`, only `n=1` is supported.

            quality: The quality of the image that will be generated. Defaults to `standard`.
                Must be one of `standard` or `hd`. `hd` creates images with finer
                details and greater consistency across the image. This param is only supported
                for `dall-e-3`.

            size: The size of the generated images. Defaults to `1024x1024`.
                Must be one of `256x256`, `512x512`, or `1024x1024` for `dall-e-2`.
                Must be one of `1024x1024`, `1792x1024`, or `1024x1792` for `dall-e-3` models.

            style: The style of the generated images. Defaults to `vivid`.
                Must be one of `vivid` or `natural`.
                Vivid causes the model to lean towards generating hyper-real and dramatic images.
                Natural causes the model to produce more natural, less hyper-real looking images.
                This param is only supported for `dall-e-3`.

            timeout: Override the client-level default timeout for this request, in seconds. Defaults to `None`.

            download: If `True`, the image will be downloaded to the cache directory. Defaults to `True`.
        """
        if size not in valid_sizes[model]:
            raise Exception(f"Invalid size for {model}: {size}")

        response = self.client.images.generate(
            prompt=text,
            n=num_images,
            model=model,
            quality=quality,
            size=size,
            response_format="b64_json" if download else "url",
            style=style,
            timeout=timeout,
        )
        if download:
            image_bytes = response.data[0].b64_json
            ext = get_extension(image_bytes)
            filename = f"{time.time()}.{ext}"

            return (self.save_base64_image(image_bytes, filename),)

        return response.data[0].url
