"""Mock utils."""

import re
from typing import List, Optional, Set

from llama_index.core.indices.keyword_table.utils import simple_extract_keywords


def mock_tokenizer(text: str) -> List[str]:
    """Mock tokenizer."""
    tokens = re.split(r"[ \n]", text)  # split by space or newline
    result = []
    for token in tokens:
        if token.strip() == "":
            continue
        result.append(token.strip())
    return result


def mock_extract_keywords(
    text_chunk: str, max_keywords: Optional[int] = None, filter_stopwords: bool = True
) -> Set[str]:
    """
    Extract keywords (mock).

    Same as simple_extract_keywords but without filtering stopwords.

    """
    return simple_extract_keywords(
        text_chunk, max_keywords=max_keywords, filter_stopwords=False
    )


def mock_extract_keywords_response(
    text_chunk: str, max_keywords: Optional[int] = None, filter_stopwords: bool = True
) -> str:
    """
    Extract keywords mock response.

    Same as simple_extract_keywords but without filtering stopwords.

    """
    return ",".join(
        simple_extract_keywords(
            text_chunk, max_keywords=max_keywords, filter_stopwords=False
        )
    )


def mock_extract_kg_triplets_response(
    text_chunk: str, max_triplets: Optional[int] = None
) -> str:
    """Generate 1 or more fake triplets."""
    response = ""
    if max_triplets is not None:
        for i in range(max_triplets):
            response += "(This is, a mock, triplet)\n"
    else:
        response += "(This is, a mock, triplet)\n"

    return response
