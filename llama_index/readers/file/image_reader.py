"""Image parser.

Contains parsers for image files.

"""

import re
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document, ImageDocument


class ImageReader(BaseReader):
    """Image parser.

    Extract text from images using DONUT.

    """

    def __init__(
        self,
        parser_config: Optional[Dict] = None,
        keep_image: bool = False,
        parse_text: bool = True,
    ):
        """Init parser."""
        if parser_config is None and parse_text:
            try:
                import sentencepiece  # noqa: F401
                import torch  # noqa: F401
                from PIL import Image  # noqa: F401
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

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        from PIL import Image

        from llama_index.img_utils import img_2_b64

        # load document image
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
            import torch

            assert self._parser_config is not None
            model = self._parser_config["model"]
            processor = self._parser_config["processor"]

            device = "cuda" if torch.cuda.is_available() else "cpu"
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

        return [ImageDocument(text=text_str, image=image_str, extra_info=extra_info)]
