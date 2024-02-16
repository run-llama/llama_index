from typing import Dict, Type

from llama_index.legacy.embeddings.base import BaseEmbedding
from llama_index.legacy.embeddings.google import GoogleUnivSentEncoderEmbedding
from llama_index.legacy.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from llama_index.legacy.embeddings.openai import OpenAIEmbedding
from llama_index.legacy.embeddings.text_embeddings_inference import (
    TextEmbeddingsInference,
)
from llama_index.legacy.embeddings.utils import resolve_embed_model
from llama_index.legacy.token_counter.mock_embed_model import MockEmbedding

RECOGNIZED_EMBEDDINGS: Dict[str, Type[BaseEmbedding]] = {
    GoogleUnivSentEncoderEmbedding.class_name(): GoogleUnivSentEncoderEmbedding,
    OpenAIEmbedding.class_name(): OpenAIEmbedding,
    LangchainEmbedding.class_name(): LangchainEmbedding,
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

    # special handling for LangchainEmbedding
    # it can be any local model technially
    if name == LangchainEmbedding.class_name():
        local_name = data.get("model_name", None)
        if local_name is not None:
            return resolve_embed_model("local:" + local_name)
        else:
            raise ValueError("LangchainEmbedding requires a model_name")

    return RECOGNIZED_EMBEDDINGS[name].from_dict(data)
