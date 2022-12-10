"""General utils functions."""

import uuid
from typing import Any, Callable, List, Optional, Set

import nltk
from transformers import GPT2TokenizerFast


class GlobalsHelper:
    """Helper to retrieve globals.

    Helpful for global caching of certain variables that can be expensive to load.
    (e.g. tokenization)

    """

    _tokenizer: Optional[GPT2TokenizerFast] = None
    _stopwords: Optional[List[str]] = None

    @property
    def tokenizer(self) -> GPT2TokenizerFast:
        """Get tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
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


def llm_token_counter(method_name_str: str) -> Callable:
    """
    Use this as a decorator for methods in index/query classes that make calls to LLMs.

    Do not use this on abstract methods.

    For example, if you do
    ```
    class GPTTreeIndexBuilder:
        ...
        @llm_token_counter("GPTTreeIndexBuilder.build_from_text")
        def build_from_text(self, documents: Sequence[BaseDocument]) -> IndexGraph:
            ...
    ```

    Then after you run `build_from_text()`, it will print the output in the form below:

    ```
    [GPTTreeIndexBuilder.build_from_text] Total token usage: <some-number> tokens
    ```
    """

    def wrap(f: Callable) -> Callable:
        def wrapped_f(_self: Any, *args: Any, **kwargs: Any) -> Any:
            start_token_ct = _self._llm_predictor.total_tokens_used
            f_return_val = f(_self, *args, **kwargs)
            net_tokens = _self._llm_predictor.total_tokens_used - start_token_ct
            print(f"> [{method_name_str}] Total token usage: {net_tokens} tokens")

            return f_return_val

        return wrapped_f

    return wrap
