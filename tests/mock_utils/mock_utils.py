"""Mock utils."""

from typing import List, Optional, Set

from gpt_index.indices.keyword_table.utils import simple_extract_keywords


def mock_tokenizer(text: str) -> List[str]:
    """Mock tokenizer."""
    tokens = text.split(" ")
    result = []
    for token in tokens:
        if token.strip() == "":
            continue
        result.append(token.strip())
    return result


def mock_extract_keywords(
    text_chunk: str, max_keywords: Optional[int] = None, filter_stopwords: bool = True
) -> Set[str]:
    """Extract keywords (mock).

    Same as simple_extract_keywords but without filtering stopwords.

    """
    return simple_extract_keywords(
        text_chunk, max_keywords=max_keywords, filter_stopwords=False
    )
