"""General utils functions."""

import os
import random
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from itertools import islice
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Type,
    Union,
    cast,
)


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
            tiktoken_import_err = (
                "`tiktoken` package not found, please run `pip install tiktoken`"
            )
            try:
                import tiktoken
            except ImportError:
                raise ImportError(tiktoken_import_err)
            enc = tiktoken.get_encoding("gpt2")
            self._tokenizer = cast(Callable[[str], List], enc.encode)
            self._tokenizer = partial(self._tokenizer, allowed_special="all")
        return self._tokenizer  # type: ignore

    @property
    def stopwords(self) -> List[str]:
        """Get stopwords."""
        if self._stopwords is None:
            try:
                import nltk
                from nltk.corpus import stopwords
            except ImportError:
                raise ImportError(
                    "`nltk` package not found, please run `pip install nltk`"
                )

            from llama_index.utils import get_cache_dir

            cache_dir = get_cache_dir()
            nltk_data_dir = os.environ.get("NLTK_DATA", cache_dir)

            # update env var for nltk so that it finds the data
            os.environ["NLTK_DATA"] = nltk_data_dir

            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", download_dir=nltk_data_dir)
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
    Taken from: https://tinyurl.com/2p89xymh

    """
    prev_values = {k: getattr(obj, k) for k in kwargs}
    for k, v in kwargs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in prev_values.items():
            setattr(obj, k, v)


@dataclass
class ErrorToRetry:
    """Exception types that should be retried.

    Args:
        exception_cls (Type[Exception]): Class of exception.
        check_fn (Optional[Callable[[Any]], bool]]):
            A function that takes an exception instance as input and returns
            whether to retry.

    """

    exception_cls: Type[Exception]
    check_fn: Optional[Callable[[Any], bool]] = None


def retry_on_exceptions_with_backoff(
    lambda_fn: Callable,
    errors_to_retry: List[ErrorToRetry],
    max_tries: int = 10,
    min_backoff_secs: float = 0.5,
    max_backoff_secs: float = 60.0,
) -> Any:
    """Execute lambda function with retries and exponential backoff.

    Args:
        lambda_fn (Callable): Function to be called and output we want.
        errors_to_retry (List[ErrorToRetry]): List of errors to retry.
            At least one needs to be provided.
        max_tries (int): Maximum number of tries, including the first. Defaults to 10.
        min_backoff_secs (float): Minimum amount of backoff time between attempts.
            Defaults to 0.5.
        max_backoff_secs (float): Maximum amount of backoff time between attempts.
            Defaults to 60.

    """
    if not errors_to_retry:
        raise ValueError("At least one error to retry needs to be provided")

    error_checks = {
        error_to_retry.exception_cls: error_to_retry.check_fn
        for error_to_retry in errors_to_retry
    }
    exception_class_tuples = tuple(error_checks.keys())

    backoff_secs = min_backoff_secs
    tries = 0

    while True:
        try:
            return lambda_fn()
        except exception_class_tuples as e:
            traceback.print_exc()
            tries += 1
            if tries >= max_tries:
                raise
            check_fn = error_checks.get(e.__class__)
            if check_fn and not check_fn(e):
                raise
            time.sleep(backoff_secs)
            backoff_secs = min(backoff_secs * 2, max_backoff_secs)


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length."""
    return text[: max_length - 3] + "..."


def iter_batch(iterable: Union[Iterable, Generator], size: int) -> Iterable:
    """Iterate over an iterable in batches.

    >>> list(iter_batch([1,2,3,4,5], 3))
    [[1, 2, 3], [4, 5]]
    """
    source_iter = iter(iterable)
    while source_iter:
        b = list(islice(source_iter, size))
        if len(b) == 0:
            break
        yield b


def concat_dirs(dir1: str, dir2: str) -> str:
    """
    Concat dir1 and dir2 while avoiding backslashes when running on windows.
    os.path.join(dir1,dir2) will add a backslash before dir2 if dir1 does not
    end with a slash, so we make sure it does.
    """
    dir1 += "/" if dir1[-1] != "/" else ""
    return os.path.join(dir1, dir2)


def get_tqdm_iterable(items: Iterable, show_progress: bool, desc: str) -> Iterable:
    """
    Optionally get a tqdm iterable. Ensures tqdm.auto is used.
    """
    _iterator = items
    if show_progress:
        try:
            from tqdm.auto import tqdm

            return tqdm(items, desc=desc)
        except ImportError:
            pass
    return _iterator


def count_tokens(text: str) -> int:
    tokens = globals_helper.tokenizer(text)
    return len(tokens)


def get_transformer_tokenizer_fn(model_name: str) -> Callable[[str], List[str]]:
    """
    Args:
        model_name(str): the model name of the tokenizer.
                        For instance, fxmarty/tiny-llama-fast-tokenizer
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ValueError(
            "`transformers` package not found, please run `pip install transformers`"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer.tokenize


def get_cache_dir() -> str:
    """Locate a platform-appropriate cache directory for llama_index,
    and create it if it doesn't yet exist
    """
    # User override
    if "LLAMA_INDEX_CACHE_DIR" in os.environ:
        path = Path(os.environ["LLAMA_INDEX_CACHE_DIR"])

    # Linux, Unix, AIX, etc.
    elif os.name == "posix" and sys.platform != "darwin":
        # use ~/.cache if empty OR not set
        base = os.path.expanduser("~/.cache")
        path = Path(base, "llama_index")

    # Mac OS
    elif sys.platform == "darwin":
        path = Path(os.path.expanduser("~"), "Library/Caches/llama_index")

    # Windows (hopefully)
    else:
        local = os.environ.get("LOCALAPPDATA", None) or os.path.expanduser(
            "~\\AppData\\Local"
        )
        path = Path(local, "llama_index")

    if not os.path.exists(path):
        os.makedirs(path)
    return str(path)


# Sample text from llama_index's readme
SAMPLE_TEXT = """
Context
LLMs are a phenomenonal piece of technology for knowledge generation and reasoning. 
They are pre-trained on large amounts of publicly available data.
How do we best augment LLMs with our own private data?
We need a comprehensive toolkit to help perform this data augmentation for LLMs.

Proposed Solution
That's where LlamaIndex comes in. LlamaIndex is a "data framework" to help 
you build LLM  apps. It provides the following tools:

Offers data connectors to ingest your existing data sources and data formats 
(APIs, PDFs, docs, SQL, etc.)
Provides ways to structure your data (indices, graphs) so that this data can be 
easily used with LLMs.
Provides an advanced retrieval/query interface over your data: 
Feed in any LLM input prompt, get back retrieved context and knowledge-augmented output.
Allows easy integrations with your outer application framework 
(e.g. with LangChain, Flask, Docker, ChatGPT, anything else).
LlamaIndex provides tools for both beginner users and advanced users. 
Our high-level API allows beginner users to use LlamaIndex to ingest and 
query their data in 5 lines of code. Our lower-level APIs allow advanced users to 
customize and extend any module (data connectors, indices, retrievers, query engines, 
reranking modules), to fit their needs.
"""
