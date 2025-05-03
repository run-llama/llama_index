from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, ImageDocument
from llama_index.core.utils import infer_torch_device


class ImageVisionLLMReader(BaseReader):
    """
    Image parser.

    Caption image using Blip2 (a multimodal VisionLLM similar to GPT4).

    """

    def __init__(
        self,
        parser_config: Optional[Dict] = None,
        keep_image: bool = False,
        prompt: str = "Question: describe what you see in this image. Answer:",
    ):
        """Init params."""
        if parser_config is None:
            try:
                import sentencepiece  # noqa
                import torch
                from PIL import Image  # noqa
                from transformers import Blip2ForConditionalGeneration, Blip2Processor
            except ImportError:
                raise ImportError(
                    "Please install extra dependencies that are required for "
                    "the ImageCaptionReader: "
                    "`pip install torch transformers sentencepiece Pillow`"
                )

            self._torch = torch
            self._torch_imported = True

            device = infer_torch_device()
            dtype = (
                self._torch.float16
                if self._torch.cuda.is_available()
                else self._torch.float32
            )
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", torch_dtype=dtype
            )
            parser_config = {
                "processor": processor,
                "model": model,
                "device": device,
                "dtype": dtype,
            }

        # Try to import PyTorch in order to run inference efficiently.
        self._import_torch()

        self._parser_config = parser_config
        self._keep_image = keep_image
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

        if self._torch_imported:
            # Gradients are not needed during inference. If PyTorch is
            # installed, we can instruct it to not track the gradients.
            # This reduces GPU memory usage and improves inference efficiency.
            with self._torch.no_grad():
                out = model.generate(**inputs)
        else:
            # Fallback to less efficient behavior if PyTorch is not installed.
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

    def _import_torch(self) -> None:
        self._torch = None

        try:
            import torch

            self._torch = torch
            self._torch_imported = True
        except ImportError:
            self._torch_imported = False
