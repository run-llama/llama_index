"""Mock text splitter."""

from typing import List, Optional

from gpt_index.langchain_helpers.text_splitter import TextSplit


def mock_token_splitter_newline(
    text: str, extra_info_str: Optional[str] = None
) -> List[str]:
    """Mock token splitter by newline."""
    if text == "":
        return []
    return text.split("\n")


def mock_token_splitter_newline_with_overlaps(
    text: str, extra_info_str: Optional[str]
) -> List[TextSplit]:
    """Mock token splitter by newline."""
    if text == "":
        return []
    strings = text.split("\n")
    return [TextSplit(string, 0) for string in strings]
