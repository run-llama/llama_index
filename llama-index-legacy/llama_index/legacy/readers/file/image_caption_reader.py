from pathlib import Path
from typing import Dict, List, Optional

from llama_index.legacy.readers.base import BaseReader
from llama_index.legacy.schema import Document, ImageDocument
from llama_index.legacy.utils import infer_torch_device


class ImageCaptionReader(BaseReader):
    """Image parser.

    Caption image using Blip.

    """

    def __init__(
        self,
        parser_config: Optional[Dict] = None,
        keep_image: bool = False,
        prompt: Optional[str] = None,
    ):
        """Init params."""
        if parser_config is None:
            """Init parser."""
            try:
                import sentencepiece  # noqa
                import torch
                from PIL import Image  # noqa
                from transformers import BlipForConditionalGeneration, BlipProcessor
            except ImportError:
                raise ImportError(
                    "Please install extra dependencies that are required for "
                    "the ImageCaptionReader: "
                    "`pip install torch transformers sentencepiece Pillow`"
                )

            device = infer_torch_device()
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            )
            model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large", torch_dtype=dtype
            )

            parser_config = {
                "processor": processor,
                "model": model,
                "device": device,
                "dtype": dtype,
            }

        self._parser_config = parser_config
        self._keep_image = keep_image
        self._prompt = prompt

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        from PIL import Image

        from llama_index.legacy.img_utils import img_2_b64

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

        out = model.generate(**inputs)
        text_str = processor.decode(out[0], skip_special_tokens=True)

        return [
            ImageDocument(
                text=text_str,
                image=image_str,
                image_path=str(file),
                metadata=extra_info or {},
            )
        ]
