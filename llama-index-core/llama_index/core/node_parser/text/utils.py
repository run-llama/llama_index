import logging
from typing import Any, Callable, List

from llama_index.core.node_parser.interface import TextSplitter

logger = logging.getLogger(__name__)


def truncate_text(text: str, text_splitter: TextSplitter) -> str:
    """Truncate text to fit within the chunk size.

    Args:
        text (str): The text to truncate.
        text_splitter (TextSplitter): The splitter to use for chunking.

    Returns:
        str: The first chunk of the split text that fits within the chunk size.
    """
    chunks = text_splitter.split_text(text)
    return chunks[0]


def split_text_keep_separator(text: str, separator: str) -> List[str]:
    """Split text with separator and keep the separator at the end of each split.

    Args:
        text (str): The text to split.
        separator (str): The separator to split on.

    Returns:
        List[str]: List of text segments with separators preserved at the end of each split.
    """
    parts = text.split(separator)
    result = [separator + s if i > 0 else s for i, s in enumerate(parts)]
    return [s for s in result if s]


def split_by_sep(sep: str, keep_sep: bool = True) -> Callable[[str], List[str]]:
    """Create a function that splits text by a separator.

    Args:
        sep (str): The separator to split on.
        keep_sep (bool, optional): Whether to keep the separator in the output. Defaults to True.

    Returns:
        Callable[[str], List[str]]: A function that takes a string and returns a list of split strings.
    """
    if keep_sep:
        return lambda text: split_text_keep_separator(text, sep)
    else:
        return lambda text: text.split(sep)


def split_by_char() -> Callable[[str], List[str]]:
    """Create a function that splits text into individual characters.

    Returns:
        Callable[[str], List[str]]: A function that takes a string and returns a list of individual characters.
    """
    return lambda text: list(text)


def split_by_sentence_tokenizer_internal(text: str, tokenizer: Any) -> List[str]:
    """Get the spans and then return the sentences.

    Using the start index of each span
    Instead of using end, use the start of the next span if available
    """
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


def split_by_sentence_tokenizer() -> Callable[[str], List[str]]:
    import nltk

    tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    return lambda text: split_by_sentence_tokenizer_internal(text, tokenizer)


def split_by_regex(regex: str) -> Callable[[str], List[str]]:
    """Create a function that splits text using a regular expression pattern.

    Args:
        regex (str): The regular expression pattern to use for splitting.

    Returns:
        Callable[[str], List[str]]: A function that takes a string and returns a list of matches based on the regex pattern.
    """
    import re

    return lambda text: re.findall(regex, text)


def split_by_phrase_regex() -> Callable[[str], List[str]]:
    """Split text by phrase regex.

    This regular expression will split the sentences into phrases,
    where each phrase is a sequence of one or more non-comma,
    non-period, and non-semicolon characters, followed by an optional comma,
    period, or semicolon. The regular expression will also capture the
    delimiters themselves as separate items in the list of phrases.
    """
    regex = "[^,.;。]+[,.;。]?"
    return split_by_regex(regex)
