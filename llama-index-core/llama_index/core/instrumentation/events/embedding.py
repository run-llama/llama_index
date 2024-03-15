from typing import List

from llama_index.core.instrumentation.events.base import BaseEvent


class EmbeddingStartEvent(BaseEvent):
    model_dict: dict

    @classmethod
    def class_name(cls):
        """Class name."""
        return "EmbeddingStartEvent"


class EmbeddingEndEvent(BaseEvent):
    chunks: List[str]
    embeddings: List[List[float]]

    @classmethod
    def class_name(cls):
        """Class name."""
        return "EmbeddingEndEvent"
