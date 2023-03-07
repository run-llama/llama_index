"""Image parser.

Contains parsers for image files.

"""

import re
from pathlib import Path
from typing import Dict

from gpt_index.readers.file.base_parser import BaseParser


class ImageParser(BaseParser):
    """Image parser.

    Extract text from images using DONUT.

    """

    def _init_parser(self) -> Dict:
        """Init parser."""
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ValueError("install pytorch to use the model")
        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
        except ImportError:
            raise ValueError("transformers is required for using DONUT model.")
        try:
            import sentencepiece  # noqa: F401
        except ImportError:
            raise ValueError("sentencepiece is required for using DONUT model.")
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            raise ValueError(
                "PIL is required to read image files." "Please run `pip install Pillow`"
            )

        processor = DonutProcessor.from_pretrained(
            "naver-clova-ix/donut-base-finetuned-cord-v2"
        )
        model = VisionEncoderDecoderModel.from_pretrained(
            "naver-clova-ix/donut-base-finetuned-cord-v2"
        )
        return {"processor": processor, "model": model}

    def parse_file(self, file: Path, errors: str = "ignore") -> str:
        """Parse file."""
        import torch
        from PIL import Image

        model = self.parser_config["model"]
        processor = self.parser_config["processor"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        # load document image
        image = Image.open(file)
        if image.mode != "RGB":
            image = image.convert("RGB")

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
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

        return sequence
