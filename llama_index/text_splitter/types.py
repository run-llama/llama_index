"""Text splitter implementations."""
from abc import ABC, abstractmethod
from typing import List, Union

from llama_index.bridge.langchain import TextSplitter as LC_TextSplitter
from llama_index.schema import BaseComponent


class TextSplitter(ABC, BaseComponent):
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        ...

    def split_texts(self, texts: List[str]) -> List[str]:
        nested_texts = [self.split_text(text) for text in texts]
        return [item for sublist in nested_texts for item in sublist]


class MetadataAwareTextSplitter(TextSplitter):
    @abstractmethod
    def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
        ...

    def split_texts_metadata_aware(
        self, texts: List[str], metadata_strs: List[str]
    ) -> List[str]:
        if len(texts) != len(metadata_strs):
            raise ValueError("Texts and metadata_strs must have the same length")
        nested_texts = [
            self.split_text_metadata_aware(text, metadata)
            for text, metadata in zip(texts, metadata_strs)
        ]
        return [item for sublist in nested_texts for item in sublist]


SplitterType = Union[TextSplitter, LC_TextSplitter]
