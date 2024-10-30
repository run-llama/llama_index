from typing import Dict
import logging

logger = logging.getLogger(__name__)


def default_sparse_encoder(text: str) -> Dict[int, int]:
    try:
        from transformers import BertTokenizer
        from collections import Counter
    except ImportError:
        raise ImportError(
            "Could not import transformers library. "
            'Please install transformers with `pip install "transformers"`'
        )

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", clean_up_tokenization_spaces=False
    )
    tokenized_text = tokenizer(text, padding=True, truncation=True, max_length=512)[
        "input_ids"
    ]
    return dict(Counter(tokenized_text))
