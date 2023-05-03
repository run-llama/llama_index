"""Image parser.

Contains parsers for image files.

"""

import re
from pathlib import Path
from typing import Dict, Optional

from llama_index.readers.file.base_parser import BaseParser, ImageParserOutput


class ImageParser(BaseParser):
    """Image parser.

    Extract text from images using DONUT.

    """

    def __init__(
        self,
        parser_config: Optional[Dict] = None,
        keep_image: bool = False,
        parse_text: bool = True,
    ):
        """Init params."""
        self._parser_config = parser_config
        self._keep_image = keep_image
        self._parse_text = parse_text

    def _init_parser(self) -> Dict:
        """Init parser."""
        if not self._parse_text:
            return {}

        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "install pytorch to use the model: " "`pip install torch`"
            )
        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
        except ImportError:
            raise ImportError(
                "transformers is required for using DONUT model: "
                "`pip install transformers`"
            )
        try:
            import sentencepiece  # noqa: F401
        except ImportError:
            raise ImportError(
                "sentencepiece is required for using DONUT model: "
                "`pip install sentencepiece`"
            )
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            raise ImportError(
                "PIL is required to read image files: " "`pip install Pillow`"
            )

        processor = DonutProcessor.from_pretrained(
            "naver-clova-ix/donut-base-finetuned-cord-v2"
        )
        model = VisionEncoderDecoderModel.from_pretrained(
            "naver-clova-ix/donut-base-finetuned-cord-v2"
        )
        return {"processor": processor, "model": model}

    def parse_file(self, file: Path, errors: str = "ignore") -> ImageParserOutput:
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

            model = self.parser_config["model"]
            processor = self.parser_config["processor"]

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

        return ImageParserOutput(
            text=text_str,
            image=image_str,
        )
