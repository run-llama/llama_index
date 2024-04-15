from typing import List, Dict


def default_sparse_encoder(texts: List[str]) -> List[Dict[int, int]]:
    try:
        from transformers import BertTokenizer
        from collections import Counter
    except ImportError:
        raise ImportError(
            "Could not import transformers library. "
            'Please install transformers with `pip install "transformers"`'
        )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    results = []
    for text in texts:
        tokenized_text = tokenizer(text, padding=True, truncation=True, max_length=512)[
            "input_ids"
        ]
        sparse_encoding = dict(Counter(tokenized_text))
        results.append(sparse_encoding)
    return results
