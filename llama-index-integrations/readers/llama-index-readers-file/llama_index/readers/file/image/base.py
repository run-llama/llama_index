"""Image parser.

Contains parsers for image files.

"""

import re
from pathlib import Path
from typing import Dict, List, Optional, cast, Any
from fsspec import AbstractFileSystem

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, ImageDocument
from llama_index.core.utils import infer_torch_device


class ImageReader(BaseReader):
    """Image parser.

    Extract text from images using DONUT or pytesseract.

    """

    def __init__(
        self,
        parser_config: Optional[Dict] = None,
        keep_image: bool = False,
        parse_text: bool = False,
        text_type: str = "text",
        pytesseract_model_kwargs: Dict[str, Any] = {},
    ):
        """Init parser."""
        self._text_type = text_type
        if parser_config is None and parse_text:
            if text_type == "plain_text":
                try:
                    import pytesseract
                except ImportError:
                    raise ImportError(
                        "Please install extra dependencies that are required for "
                        "the ImageReader when text_type is 'plain_text': "
                        "`pip install pytesseract`"
                    )
                processor = None
                model = pytesseract
            else:
                try:
                    import sentencepiece  # noqa
                    import torch  # noqa
                    from PIL import Image  # noqa
                    from transformers import DonutProcessor, VisionEncoderDecoderModel
                except ImportError:
                    raise ImportError(
                        "Please install extra dependencies that are required for "
                        "the ImageCaptionReader: "
                        "`pip install torch transformers sentencepiece Pillow`"
                    )

                processor = DonutProcessor.from_pretrained(
                    "naver-clova-ix/donut-base-finetuned-cord-v2"
                )
                model = VisionEncoderDecoderModel.from_pretrained(
                    "naver-clova-ix/donut-base-finetuned-cord-v2"
                )
            parser_config = {"processor": processor, "model": model}

        self._parser_config = parser_config
        self._keep_image = keep_image
        self._parse_text = parse_text
        self._pytesseract_model_kwargs = pytesseract_model_kwargs

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        from llama_index.core.img_utils import img_2_b64
        from PIL import Image

        # load document image
        if fs:
            with fs.open(path=file) as f:
                image = Image.open(f.read())
        else:
            image = Image.open(file)

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Encode image into base64 string and keep in document
        image_str: Optional[str] = None
        if self._keep_image:
            image_str = img_2_b64(image)

        # Parse image into text
        text_str: str = ""
        if self._parse_text:
            assert self._parser_config is not None
            model = self._parser_config["model"]
            processor = self._parser_config["processor"]

            if processor:
                device = infer_torch_device()
                model.to(device)

                # prepare decoder inputs
                task_prompt = "<s_cord-v2>"
                decoder_input_ids = processor.tokenizer(
                    task_prompt, add_special_tokens=False, return_tensors="pt"
                ).input_ids

                pixel_values = processor(image, return_tensors="pt").pixel_values

                outputs = model.generate(
                    pixel_values.to(device),
                    decoder_input_ids=decoder_input_ids.to(device),
                    max_length=model.decoder.config.max_position_embeddings,
                    early_stopping=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=3,
                    bad_words_ids=[[processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )

                sequence = processor.batch_decode(outputs.sequences)[0]
                sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
                    processor.tokenizer.pad_token, ""
                )
                # remove first task start token
                text_str = re.sub(r"<.*?>", "", sequence, count=1).strip()
            else:
                import pytesseract

                model = cast(pytesseract, self._parser_config["model"])
                text_str = model.image_to_string(
                    image, **self._pytesseract_model_kwargs
                )

        return [
            ImageDocument(
                text=text_str,
                image=image_str,
                image_path=str(file),
                metadata=extra_info or {},
            )
        ]
