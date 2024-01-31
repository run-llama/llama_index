from typing import Dict, Type

from llama_index.core.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.embeddings.openai import OpenAIEmbedding
from llama_index.core.embeddings.text_embeddings_inference import (
    TextEmbeddingsInference,
)

RECOGNIZED_EMBEDDINGS: Dict[str, Type[BaseEmbedding]] = {
    OpenAIEmbedding.class_name(): OpenAIEmbedding,
    MockEmbedding.class_name(): MockEmbedding,
    HuggingFaceEmbedding.class_name(): HuggingFaceEmbedding,
    TextEmbeddingsInference.class_name(): TextEmbeddingsInference,
    OpenAIEmbedding.class_name(): OpenAIEmbedding,
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
