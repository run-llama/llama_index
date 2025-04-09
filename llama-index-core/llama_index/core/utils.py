"""General utils functions."""

import asyncio
import base64
import os
import random
import requests
import sys
import time
import traceback
import uuid
from binascii import Error as BinasciiError
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, wraps
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    Union,
    runtime_checkable,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from nltk.tokenize import PunktSentenceTokenizer


class GlobalsHelper:
    """Helper to retrieve globals with asynchronous NLTK data loading."""

    _stopwords: Optional[List[str]] = None
    _punkt_tokenizer: Optional["PunktSentenceTokenizer"] = None
    _nltk_data_dir: Optional[str] = None

    def wait_for_nltk_check(self) -> None:
        """Initialize NLTK data download."""
        from nltk.data import path as nltk_path

        # Set up NLTK data directory
        self._nltk_data_dir = os.environ.get(
            "NLTK_DATA",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "_static/nltk_cache",
            ),
        )

        # Ensure the directory exists
        os.makedirs(self._nltk_data_dir, exist_ok=True)

        # Add to NLTK path if not already present
        if self._nltk_data_dir not in nltk_path:
            nltk_path.append(self._nltk_data_dir)

        # Start downloading NLTK data / confirming it's available
        self._download_nltk_data()

    def _download_nltk_data(self) -> None:
        """Download NLTK data packages in the background."""
        from nltk.data import find as nltk_find
        from nltk import download

        try:
            # Download stopwords
            try:
                nltk_find("corpora/stopwords", paths=[self._nltk_data_dir])
            except LookupError:
                download("stopwords", download_dir=self._nltk_data_dir, quiet=True)

            # Download punkt tokenizer
            try:
                nltk_find("tokenizers/punkt_tab", paths=[self._nltk_data_dir])
            except LookupError:
                download("punkt_tab", download_dir=self._nltk_data_dir, quiet=True)

        except Exception as e:
            print(f"NLTK download error: {e}")

    @property
    def stopwords(self) -> List[str]:
        """Get stopwords, ensuring data is downloaded."""
        if self._stopwords is None:
            # Wait for stopwords to be available
            self.wait_for_nltk_check()

            from nltk.corpus import stopwords
            from nltk.tokenize import PunktSentenceTokenizer

            self._stopwords = stopwords.words("english")
            self._punkt_tokenizer = PunktSentenceTokenizer()

        return self._stopwords

    @property
    def punkt_tokenizer(self) -> "PunktSentenceTokenizer":
        """Get punkt tokenizer, ensuring data is downloaded."""
        if self._punkt_tokenizer is None:
            # Wait for punkt to be available
            self.wait_for_nltk_check()

            from nltk.corpus import stopwords
            from nltk.tokenize import PunktSentenceTokenizer

            self._punkt_tokenizer = PunktSentenceTokenizer()
            self._stopwords = stopwords.words("english")

        return self._punkt_tokenizer


globals_helper = GlobalsHelper()


# Global Tokenizer
@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text: str, *args: Any, **kwargs: Any) -> List[Any]:
        ...


def set_global_tokenizer(tokenizer: Union[Tokenizer, Callable[[str], list]]) -> None:
    import llama_index.core

    if isinstance(tokenizer, Tokenizer):
        llama_index.core.global_tokenizer = tokenizer.encode
    else:
        llama_index.core.global_tokenizer = tokenizer


def get_tokenizer() -> Callable[[str], List]:
    import llama_index.core

    if llama_index.core.global_tokenizer is None:
        tiktoken_import_err = (
            "`tiktoken` package not found, please run `pip install tiktoken`"
        )
        try:
            import tiktoken
        except ImportError:
            raise ImportError(tiktoken_import_err)

        # set tokenizer cache temporarily
        should_revert = False
        if "TIKTOKEN_CACHE_DIR" not in os.environ:
            should_revert = True
            os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "_static/tiktoken_cache",
            )

        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokenizer = partial(enc.encode, allowed_special="all")
        set_global_tokenizer(tokenizer)

        if should_revert:
            del os.environ["TIKTOKEN_CACHE_DIR"]

    assert llama_index.core.global_tokenizer is not None
    return llama_index.core.global_tokenizer


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


async def aretry_on_exceptions_with_backoff(
    async_fn: Callable,
    errors_to_retry: List[ErrorToRetry],
    max_tries: int = 10,
    min_backoff_secs: float = 0.5,
    max_backoff_secs: float = 60.0,
) -> Any:
    """Execute lambda function with retries and exponential backoff.

    Args:
        async_fn (Callable): Async Function to be called and output we want.
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
            return await async_fn()
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


def get_retry_on_exceptions_with_backoff_decorator(
    *retry_args: Any, **retry_kwargs: Any
) -> Callable:
    """Return a decorator that retries with exponential backoff on provided exceptions."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*func_args: Any, **func_kwargs: Any) -> Any:
            return retry_on_exceptions_with_backoff(
                lambda: func(*func_args, **func_kwargs), *retry_args, **retry_kwargs
            )

        @wraps(func)
        async def awrapper(*func_args: Any, **func_kwargs: Any) -> Any:
            async def foo() -> Any:
                return await func(*func_args, **func_kwargs)

            return await aretry_on_exceptions_with_backoff(
                foo, *retry_args, **retry_kwargs
            )

        return awrapper if asyncio.iscoroutinefunction(func) else wrapper

    return decorator


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
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


def concat_dirs(dirname: str, basename: str) -> str:
    """
    Append basename to dirname, avoiding backslashes when running on windows.

    os.path.join(dirname, basename) will add a backslash before dirname if
    basename does not end with a slash, so we make sure it does.
    """
    dirname += "/" if dirname[-1] != "/" else ""
    return os.path.join(dirname, basename)


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
    tokenizer = get_tokenizer()
    tokens = tokenizer(text)
    return len(tokens)


def get_transformer_tokenizer_fn(model_name: str) -> Callable[[str], List[str]]:
    """
    Args:
        model_name(str): the model name of the tokenizer.
                        For instance, fxmarty/tiny-llama-fast-tokenizer.
    """
    try:
        from transformers import AutoTokenizer  # pants: no-infer-dep
    except ImportError:
        raise ValueError(
            "`transformers` package not found, please run `pip install transformers`"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer.tokenize


def get_cache_dir() -> str:
    """Locate a platform-appropriate cache directory for llama_index,
    and create it if it doesn't yet exist.
    """
    # User override
    if "LLAMA_INDEX_CACHE_DIR" in os.environ:
        path = Path(os.environ["LLAMA_INDEX_CACHE_DIR"])

    # Linux, Unix, AIX, etc.
    elif os.name == "posix" and sys.platform != "darwin":
        path = Path("/tmp/llama_index")

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
        os.makedirs(
            path, exist_ok=True
        )  # prevents https://github.com/jerryjliu/llama_index/issues/7362
    return str(path)


def add_sync_version(func: Any) -> Any:
    """Decorator for adding sync version of an async function. The sync version
    is added as a function attribute to the original function, func.

    Args:
        func(Any): the async function for which a sync variant will be built.
    """
    assert asyncio.iscoroutinefunction(func)

    @wraps(func)
    def _wrapper(*args: Any, **kwds: Any) -> Any:
        return asyncio.get_event_loop().run_until_complete(func(*args, **kwds))

    func.sync = _wrapper
    return func


# Sample text from llama_index's readme
SAMPLE_TEXT = """
Context
LLMs are a phenomenal piece of technology for knowledge generation and reasoning.
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

_LLAMA_INDEX_COLORS = {
    "llama_pink": "38;2;237;90;200",
    "llama_blue": "38;2;90;149;237",
    "llama_turquoise": "38;2;11;159;203",
    "llama_lavender": "38;2;155;135;227",
}

_ANSI_COLORS = {
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "pink": "38;5;200",
}


def get_color_mapping(
    items: List[str], use_llama_index_colors: bool = True
) -> Dict[str, str]:
    """
    Get a mapping of items to colors.

    Args:
        items (List[str]): List of items to be mapped to colors.
        use_llama_index_colors (bool, optional): Flag to indicate
        whether to use LlamaIndex colors or ANSI colors.
            Defaults to True.

    Returns:
        Dict[str, str]: Mapping of items to colors.
    """
    if use_llama_index_colors:
        color_palette = _LLAMA_INDEX_COLORS
    else:
        color_palette = _ANSI_COLORS

    colors = list(color_palette.keys())
    return {item: colors[i % len(colors)] for i, item in enumerate(items)}


def _get_colored_text(text: str, color: str) -> str:
    """
    Get the colored version of the input text.

    Args:
        text (str): Input text.
        color (str): Color to be applied to the text.

    Returns:
        str: Colored version of the input text.
    """
    all_colors = {**_LLAMA_INDEX_COLORS, **_ANSI_COLORS}

    if color not in all_colors:
        return f"\033[1;3m{text}\033[0m"  # just bolded and italicized

    color = all_colors[color]

    return f"\033[1;3;{color}m{text}\033[0m"


def print_text(text: str, color: Optional[str] = None, end: str = "") -> None:
    """
    Print the text with the specified color.

    Args:
        text (str): Text to be printed.
        color (str, optional): Color to be applied to the text. Supported colors are:
            llama_pink, llama_blue, llama_turquoise, llama_lavender,
            red, green, yellow, blue, magenta, cyan, pink.
        end (str, optional): String appended after the last character of the text.

    Returns:
        None
    """
    text_to_print = _get_colored_text(text, color) if color is not None else text
    print(text_to_print, end=end)


def infer_torch_device() -> str:
    """Infer the input to torch.device."""
    try:
        has_cuda = torch.cuda.is_available()
    except NameError:
        import torch  # pants: no-infer-dep

        has_cuda = torch.cuda.is_available()
    if has_cuda:
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def unit_generator(x: Any) -> Generator[Any, None, None]:
    """A function that returns a generator of a single element.

    Args:
        x (Any): the element to build yield

    Yields:
        Any: the single element
    """
    yield x


async def async_unit_generator(x: Any) -> AsyncGenerator[Any, None]:
    """A function that returns a generator of a single element.

    Args:
        x (Any): the element to build yield

    Yields:
        Any: the single element
    """
    yield x


def resolve_binary(
    raw_bytes: Optional[bytes] = None,
    path: Optional[Union[str, Path]] = None,
    url: Optional[str] = None,
    as_base64: bool = False,
) -> BytesIO:
    """Resolve binary data from various sources into a BytesIO object.

    Args:
        raw_bytes: Raw bytes data
        path: File path to read bytes from
        url: URL to fetch bytes from
        as_base64: Whether to base64 encode the output bytes

    Returns:
        BytesIO object containing the binary data

    Raises:
        ValueError: If no valid source is provided
    """
    if raw_bytes is not None:
        # check if raw_bytes is base64 encoded
        try:
            decoded_bytes = base64.b64decode(raw_bytes)
        except Exception:
            decoded_bytes = raw_bytes

        try:
            # Check if raw_bytes is already base64 encoded.
            # b64decode() can succeed on random binary data, so we
            # pass verify=True to make sure it's not a false positive
            decoded_bytes = base64.b64decode(raw_bytes, validate=True)
        except BinasciiError:
            # b64decode failed, leave as is
            decoded_bytes = raw_bytes

        if as_base64:
            return BytesIO(base64.b64encode(decoded_bytes))
        return BytesIO(decoded_bytes)

    elif path is not None:
        path = Path(path) if isinstance(path, str) else path
        data = path.read_bytes()
        if as_base64:
            return BytesIO(base64.b64encode(data))
        return BytesIO(data)

    elif url is not None:
        headers = {
            "User-Agent": "LlamaIndex/0.0 (https://llamaindex.ai; info@llamaindex.ai) llama-index-core/0.0"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        if as_base64:
            return BytesIO(base64.b64encode(response.content))
        return BytesIO(response.content)

    raise ValueError("No valid source provided to resolve binary data!")
