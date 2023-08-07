"""Text splitter implementations."""
from typing import List, Protocol


class TextSplitter(Protocol):
    def split_text(self, text: str) -> List[str]:
        ...
