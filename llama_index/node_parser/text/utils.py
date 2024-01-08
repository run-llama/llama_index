import logging
from functools import partial
from typing import Any, Generator, List, Protocol, Tuple

from llama_index.node_parser.interface import TextSplitter

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class SplitCallable(Protocol):
    """Callable Protocol for Splitters."""

    def __call__(self, text: str) -> List[str]:
        ...


class SpanTokenizer(Protocol):
    """Callable protocol for tokenizer with span_tokenize method."""

    def span_tokenize(self, text: str) -> Generator[Tuple[int], Any, Any]:
        ...


def split_text_by_sep(text: str, sep: str) -> List[str]:
    return text.split(sep)


def truncate_text(text: str, text_splitter: TextSplitter) -> str:
    """Truncate text to fit within the chunk size."""
    chunks = text_splitter.split_text(text)
    return chunks[0]


def split_text_keep_separator(text: str, separator: str) -> List[str]:
    """Split text with separator and keep the separator at the end of each split."""
    parts = text.split(separator)
    result = [separator + s if i > 0 else s for i, s in enumerate(parts)]
    return [s for s in result if s]


def split_by_sep(sep: str, keep_sep: bool = True) -> SplitCallable:
    """Split text by separator."""
    if keep_sep:
        return partial(split_text_keep_separator, separator=sep)
    else:
        return partial(split_text_by_sep, sep=sep)


def _split_by_char(text: str) -> List[str]:
    """Split text by character helper."""
    return list(text)


def split_by_char() -> SplitCallable:
    """Split text by character."""
    return _split_by_char


def _split_by_sentence_tokenizer(tokenizer: SpanTokenizer, text: str) -> List[str]:
    # get the spans and then return the sentences
    # using the start index of each span
    # instead of using end, use the start of the next span if available
    spans = list(tokenizer.span_tokenize(text))
    sentences = []
    for i, span in enumerate(spans):
        start = span[0]
        if i < len(spans) - 1:
            end = spans[i + 1][0]
        else:
            end = len(text)
        sentences.append(text[start:end])

    return sentences


def split_by_sentence_tokenizer() -> SplitCallable:
    import nltk

    tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    return partial(_split_by_sentence_tokenizer, tokenizer=tokenizer)


def _split_by_regex(text: str, regex: str) -> List[str]:
    """Split text by regex helper."""
    import re

    return re.findall(regex, text)


def split_by_regex(regex: str) -> SplitCallable:
    """Split text by regex."""
    return partial(_split_by_regex, regex=regex)


def split_by_phrase_regex() -> SplitCallable:
    """Split text by phrase regex.

    This regular expression will split the sentences into phrases,
    where each phrase is a sequence of one or more non-comma,
    non-period, and non-semicolon characters, followed by an optional comma,
    period, or semicolon. The regular expression will also capture the
    delimiters themselves as separate items in the list of phrases.
    """
    regex = "[^,.;。]+[,.;。]?"
    return split_by_regex(regex)
