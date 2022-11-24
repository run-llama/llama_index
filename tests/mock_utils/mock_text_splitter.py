"""Mock text splitter."""

from typing import List


def mock_token_splitter_newline(text: str) -> List[str]:
    """Mock token splitter by newline."""
    return text.split("\n")
