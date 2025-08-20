from typing import Dict, List

from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.bridge.pydantic import ConfigDict


class EmbeddingStartEvent(BaseEvent):
    """
    EmbeddingStartEvent.

    Args:
        model_dict (dict): Model dictionary containing details about the embedding model.

    """

    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    model_dict: dict

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "EmbeddingStartEvent"


class EmbeddingEndEvent(BaseEvent):
    """
    EmbeddingEndEvent.

    Args:
        chunks (List[str]): List of chunks.
        embeddings (List[List[float]]): List of embeddings.

    """

    chunks: List[str]
    embeddings: List[List[float]]

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "EmbeddingEndEvent"


class SparseEmbeddingStartEvent(BaseEvent):
    """
    EmbeddingStartEvent.

    Args:
        model_dict (dict): Model dictionary containing details about the embedding model.

    """

    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    model_dict: dict

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "SparseEmbeddingStartEvent"


class SparseEmbeddingEndEvent(BaseEvent):
    """
    EmbeddingEndEvent.

    Args:
        chunks (List[str]): List of chunks.
        embeddings (List[List[float]]): List of embeddings.

    """

    chunks: List[str]
    embeddings: List[Dict[int, float]]

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "SparseEmbeddingEndEvent"
