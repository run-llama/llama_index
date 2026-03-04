from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, ImageDocument


class ImageTabularChartReader(BaseReader):
    """
    Image parser.

    Extract tabular data from a chart or figure.

    """

    def __init__(
        self,
        parser_config: Optional[Dict] = None,
        keep_image: bool = False,
        max_output_tokens=512,
        prompt: str = "Generate underlying data table of the figure below:",
    ):
        """Init params."""
        if parser_config is None:
            try:
                import torch
                from PIL import Image  # noqa: F401
                from transformers import (
                    Pix2StructForConditionalGeneration,
                    Pix2StructProcessor,
                )
            except ImportError:
                raise ImportError(
                    "Please install extra dependencies that are required for "
                    "the ImageCaptionReader: "
                    "`pip install torch transformers Pillow`"
                )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            processor = Pix2StructProcessor.from_pretrained("google/deplot")
            model = Pix2StructForConditionalGeneration.from_pretrained(
                "google/deplot", torch_dtype=dtype
            )
            parser_config = {
                "processor": processor,
                "model": model,
                "device": device,
                "dtype": dtype,
            }

        self._parser_config = parser_config
        self._keep_image = keep_image
        self._max_output_tokens = max_output_tokens
        self._prompt = prompt

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        from llama_index.core.img_utils import img_2_b64
        from PIL import Image

        # load document image
        image = Image.open(file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Encode image into base64 string and keep in document
        image_str: Optional[str] = None
        if self._keep_image:
            image_str = img_2_b64(image)

        # Parse image into text
        model = self._parser_config["model"]
        processor = self._parser_config["processor"]

        device = self._parser_config["device"]
        dtype = self._parser_config["dtype"]
        model.to(device)

        # unconditional image captioning

        inputs = processor(image, self._prompt, return_tensors="pt").to(device, dtype)

        out = model.generate(**inputs, max_new_tokens=self._max_output_tokens)
        text_str = "Figure or chart with tabular data: " + processor.decode(
            out[0], skip_special_tokens=True
        )

        return [
            ImageDocument(
                text=text_str,
                image=image_str,
                extra_info=extra_info or {},
            )
        ]
