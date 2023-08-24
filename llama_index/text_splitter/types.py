"""Text splitter implementations."""
from abc import ABC, abstractmethod
from typing import List

from pydantic.v1 import BaseModel


class TextSplitter(ABC, BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        ...


class MetadataAwareTextSplitter(TextSplitter):
    @abstractmethod
    def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
        ...
