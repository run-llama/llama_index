from typing import Dict, Type

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.mock_embed_model import MockEmbedding

RECOGNIZED_EMBEDDINGS: Dict[str, Type[BaseEmbedding]] = {
    MockEmbedding.class_name(): MockEmbedding,
}


def load_embed_model(data: dict) -> BaseEmbedding:
    """Load Embedding by name."""
    if isinstance(data, BaseEmbedding):
        return data
    name = data.get("class_name", None)
    if name is None:
        raise ValueError("Embedding loading requires a class_name")
    if name not in RECOGNIZED_EMBEDDINGS:
        raise ValueError(f"Invalid Embedding name: {name}")

    return RECOGNIZED_EMBEDDINGS[name].from_dict(data)
