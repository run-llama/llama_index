"""Text splitter implementations."""
from typing import List, Protocol, runtime_checkable


class TextSplitter(Protocol):
    def split_text(self, text: str) -> List[str]:
        ...


@runtime_checkable
class MetadataAwareTextSplitter(Protocol):
    def split_text(self, text: str) -> List[str]:
        ...

    def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
        ...