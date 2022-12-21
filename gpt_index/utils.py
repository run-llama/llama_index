"""General utils functions."""

import random
import sys
import uuid
from typing import Any, Callable, List, Optional, Set

import nltk
from transformers import GPT2TokenizerFast


class GlobalsHelper:
    """Helper to retrieve globals.

    Helpful for global caching of certain variables that can be expensive to load.
    (e.g. tokenization)

    """

    _tokenizer: Optional[Callable[[str], List]] = None
    _stopwords: Optional[List[str]] = None

    @property
    def tokenizer(self) -> Callable[[str], List]:
        """Get tokenizer."""
        if self._tokenizer is None:
            # if python version >= 3.9, then use tiktoken
            # else use GPT2TokenizerFast
            if sys.version_info >= (3, 9):
                tiktoken_import_err = (
                    "`tiktoken` package not found, please run `pip install tiktoken`"
                )
                try:
                    import tiktoken
                except ImportError:
                    raise ValueError(tiktoken_import_err)
                enc = tiktoken.get_encoding("gpt2")
                self._tokenizer = enc.encode
            else:
                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

                def tokenizer_fn(text: str) -> List:
                    return tokenizer(text)["input_ids"]

                self._tokenizer = tokenizer_fn
        return self._tokenizer

    @property
    def stopwords(self) -> List[str]:
        """Get stopwords."""
        if self._stopwords is None:
            try:
                from nltk.corpus import stopwords
            except ImportError:
                raise ValueError(
                    "`nltk` package not found, please run `pip install nltk`"
                )
            nltk.download("stopwords")
            self._stopwords = stopwords.words("english")
        return self._stopwords


globals_helper = GlobalsHelper()


def get_new_id(d: Set) -> str:
    """Get a new ID."""
    while True:
        new_id = str(uuid.uuid4())
        if new_id not in d:
            break
    return new_id


def get_new_int_id(d: Set) -> int:
    """Get a new integer ID."""
    while True:
        new_id = random.randint(0, sys.maxsize)
        if new_id not in d:
            break
    return new_id


def llm_token_counter(method_name_str: str) -> Callable:
    """
    Use this as a decorator for methods in index/query classes that make calls to LLMs.

    At the moment, this decorator can only be used on class instance methods with a
    `_llm_predictor` attribute.

    Do not use this on abstract methods.

    For example, consider the class below:
        .. code-block:: python
            class GPTTreeIndexBuilder:
            ...
            @llm_token_counter("build_from_text")
            def build_from_text(self, documents: Sequence[BaseDocument]) -> IndexGraph:
                ...

    If you run `build_from_text()`, it will print the output in the form below:

    ```
    [build_from_text] Total token usage: <some-number> tokens
    ```
    """

    def wrap(f: Callable) -> Callable:
        def wrapped_llm_predict(_self: Any, *args: Any, **kwargs: Any) -> Any:
            start_token_ct = _self._llm_predictor.total_tokens_used
            f_return_val = f(_self, *args, **kwargs)
            net_tokens = _self._llm_predictor.total_tokens_used - start_token_ct
            _self._llm_predictor.last_token_usage = net_tokens
            print(f"> [{method_name_str}] Total token usage: {net_tokens} tokens")

            return f_return_val

        return wrapped_llm_predict

    return wrap
