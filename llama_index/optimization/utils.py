from typing import Callable, List

IMPORT_ERROR_MSG = (
    "`transformers` package not found, please run `pip install transformers`"
)


def get_transformer_tokenizer_fin(model_name: str) -> Callable[[str], List[str]]:
    """
    Args:
        model_name(str): the model name of the tokenizer.
                        For instance, fxmarty/tiny-llama-fast-tokenizer
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ValueError(IMPORT_ERROR_MSG)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer.tokenize


def get_large_chinese_tokenizer_fn() -> Callable[[str], List[str]]:
    # Here gives an example of large-chinese-tokenizer
    return get_transformer_tokenizer_fin("GanymedeNil/text2vec-large-chinese")
