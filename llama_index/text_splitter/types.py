"""Text splitter implementations."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class TextSplit:
    """Text split with overlap.

    Attributes:
        text_chunk: The text string.
        num_char_overlap: The number of overlapping characters with the previous chunk.
    """

    text_chunk: str
    num_char_overlap: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class TextSplitter(Protocol):
    def split_text(self, text: str) -> List[str]:
        ...
