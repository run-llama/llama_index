"""General utils functions."""

import random
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Generator, List, Optional, Set, Type, cast

import nltk


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
                self._tokenizer = cast(Callable[[str], List], enc.encode)
            else:
                import transformers

                tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

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


@contextmanager
def temp_set_attrs(obj: Any, **kwargs: Any) -> Generator:
    """Temporary setter.

    Utility class for setting a temporary value for an attribute on a class.
    Taken from: shorturl.at/fBFQ5.

    """
    prev_values = {k: getattr(obj, k) for k in kwargs}
    for k, v in kwargs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in prev_values.items():
            setattr(obj, k, v)


def retry_on_exceptions_with_backoff(
    lambda_fn: Callable,
    exception_classes: List[Type[Exception]],
    max_tries: int = 10,
    min_backoff_secs: float = 0.5,
    max_backoff_secs: float = 60.0,
) -> Any:
    """Execute lambda function with retries and exponential backoff.

    Args:
        lambda_fn (Callable): Function to be called and output we want.
        exception_classes (List[Type[Exception]]): List of exception classes to retry.
        max_tries (int): Maximum number of tries, including the first. Defaults to 10.
        min_backoff_secs (float): Minimum amount of backoff time between attempts.
            Defaults to 0.5.
        max_backoff_secs (float): Maximum amount of backoff time between attempts.
            Defaults to 60.

    """
    exception_class_tuples = tuple(exception_classes)
    backoff_secs = min_backoff_secs
    tries = 0
    while True:
        try:
            return lambda_fn()
        except exception_class_tuples:
            traceback.print_exc()
            tries += 1
            if tries >= max_tries:
                raise
            time.sleep(backoff_secs)
            backoff_secs = min(backoff_secs * 2, max_backoff_secs)
