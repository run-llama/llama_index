from typing import Dict, Type

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.mock_embed_model import MockEmbedding

RECOGNIZED_EMBEDDINGS: Dict[str, Type[BaseEmbedding]] = {
    MockEmbedding.class_name(): MockEmbedding,
}

# conditionals for llama-cloud support
try:
    from llama_index.embeddings.openai import OpenAIEmbedding  # pants: no-infer-dep

    RECOGNIZED_EMBEDDINGS[OpenAIEmbedding.class_name()] = OpenAIEmbedding
except ImportError:
    pass

try:
    from llama_index.embeddings.azure_openai import (
        AzureOpenAIEmbedding,
    )  # pants: no-infer-dep

    RECOGNIZED_EMBEDDINGS[AzureOpenAIEmbedding.class_name()] = AzureOpenAIEmbedding
except ImportError:
    pass

try:
    from llama_index.embeddings.huggingface_api import (
        HuggingFaceInferenceAPIEmbedding,
    )  # pants: no-infer-dep

    RECOGNIZED_EMBEDDINGS[HuggingFaceInferenceAPIEmbedding.class_name()] = (
        HuggingFaceInferenceAPIEmbedding
    )
except ImportError:
    pass


def load_embed_model(data: dict) -> BaseEmbedding:
    """Load Embedding by name."""
    if isinstance(data, BaseEmbedding):
        return data
    name = data.get("class_name")
    if name is None:
        raise ValueError("Embedding loading requires a class_name")
    if name not in RECOGNIZED_EMBEDDINGS:
        raise ValueError(f"Invalid Embedding name: {name}")

    return RECOGNIZED_EMBEDDINGS[name].from_dict(data)
