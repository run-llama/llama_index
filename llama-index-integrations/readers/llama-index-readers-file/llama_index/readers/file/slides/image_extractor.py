"""
Image extraction utilities for PowerPoint slides.

Handles image captioning using vision models.
"""

import logging
import tempfile
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ImageExtractor:
    """
    Handles image extraction and captioning for PowerPoint slides.

    Uses vision transformer models for image captioning.
    """

    def __init__(self):
        """Initialize image extractor with vision models."""
        self.vision_models = None
        self._initialize_vision_models()

    def _initialize_vision_models(self) -> None:
        """Initialize vision transformer models for image captioning."""
        try:
            import torch  # noqa
            from PIL import Image  # noqa
            from transformers import (
                AutoTokenizer,
                VisionEncoderDecoderModel,
                ViTFeatureExtractor,
            )
        except ImportError:
            raise ImportError(
                "Missing required dependencies for image extraction and captioning.\n"
                "Please install the following packages:\n"
                "  pip install 'torch>=2.7.1' 'transformers<4.50' 'pillow>=11.2.1'\n\n"
                "Note: This feature requires PyTorch and transformers for AI-powered image captioning.\n"
                "If you don't need image extraction, set extract_images=False when initializing PptxReader."
            )

        model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        feature_extractor = ViTFeatureExtractor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )

        self.vision_models = {
            "feature_extractor": feature_extractor,
            "model": model,
            "tokenizer": tokenizer,
        }

    def caption_image_from_file(self, image_path: str) -> str:
        """
        Generate caption for image from file path.

        Args:
            image_path: Path to image file

        Returns:
            Image caption text

        """
        if not self.vision_models:
            raise RuntimeError(
                "Image captioning not available - vision models not loaded"
            )

        from PIL import Image
        from llama_index.core.utils import infer_torch_device

        model = self.vision_models["model"]
        feature_extractor = self.vision_models["feature_extractor"]
        tokenizer = self.vision_models["tokenizer"]

        device = infer_torch_device()
        model.to(device)

        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        pixel_values = feature_extractor(
            images=[i_image], return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, **gen_kwargs)

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return preds[0].strip()

    def extract_image_data(self, shape, slide_number: int) -> Dict[str, Any]:
        """
        Extract image data and caption from PowerPoint shape.

        Args:
            shape: PowerPoint shape containing image
            slide_number: Slide number for context

        Returns:
            Dictionary with image metadata and caption

        """
        # Use temp file approach like original code
        image_bytes = shape.image.blob
        f = tempfile.NamedTemporaryFile(
            "wb", delete=False, suffix=f".{shape.image.ext}"
        )
        try:
            f.write(image_bytes)
            f.close()
            caption = self.caption_image_from_file(f.name)
        finally:
            os.unlink(f.name)

        return {
            "type": "image",
            "format": shape.image.ext,
            "caption": caption,
            "content": caption,
        }
