from typing import Dict
import logging
import pandas as pd

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

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_text = tokenizer(text, padding=True, truncation=True, max_length=512)[
        "input_ids"
    ]
    return dict(Counter(tokenized_text))


# MATCH THE METADATA COLUMN DATA TYPE TO ITS PYTYPE
def convert_metadata_col(column, value):
    try:
        if column["type"] == "str":
            return str(value)
        elif column["type"] == "bytes":
            return value.encode("utf-8")
        elif column["type"] == "datetime64[ns]":
            return pd.to_datetime(value)
        elif column["type"] == "timedelta64[ns]":
            return pd.to_timedelta(value)
        return value.astype(column["type"])
    except Exception as e:
        logger.error(
            f"Failed to convert column {column['name']} to type {column['pytype']}: {e}"
        )
