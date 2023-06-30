from typing import Callable, List


def get_transformer_tokenizer_fin(model_name: str) -> Callable[[str], List[str]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer.tokenize


def get_large_chinese_tokenizer_fn() -> Callable[[str], List[str]]:
    return get_transformer_tokenizer_fin("GanymedeNil/text2vec-large-chinese")
