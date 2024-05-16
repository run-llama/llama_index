from typing import List, Sequence

from llama_index.core.instrumentation.events.base import BaseEvent


class EmbeddingStartEvent(BaseEvent):
    """EmbeddingStartEvent.

    Args:
        model_dict (dict): Model dictionary containing details about the embedding model.
    """

    model_dict: dict

    @classmethod
    def class_name(cls):
        """Class name."""
        return "EmbeddingStartEvent"


class EmbeddingEndEvent(BaseEvent):
    """EmbeddingEndEvent.

    Args:
        chunks (Sequence[object]): List of chunks.
        embeddings (List[List[float]]): List of embeddings.

    """

    chunks: Sequence[object]
    embeddings: List[List[float]]

    @classmethod
    def class_name(cls):
        """Class name."""
        return "EmbeddingEndEvent"
