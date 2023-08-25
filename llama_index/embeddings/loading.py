from llama_index.embeddings.base import BaseEmbedding
from llama_index.embeddings.google import GoogleUnivSentEncoderEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.embeddings.utils import resolve_embed_model
from llama_index.token_counter.mock_embed_model import MockEmbedding


def load_embed_model(data: dict) -> BaseEmbedding:
    """Load Embedding by name."""
    name = data.get("class_name", None)
    if name is None:
        raise ValueError("Embedding loading requires a class_name")

    if name == GoogleUnivSentEncoderEmbedding.__name__:
        return GoogleUnivSentEncoderEmbedding.from_dict(data)
    elif name == OpenAIEmbedding.__name__:
        return OpenAIEmbedding.from_dict(data)
    elif name == LangchainEmbedding.__name__:
        local_name = data.get("model_name", None)
        if local_name is not None:
            return resolve_embed_model("local:" + local_name)
        else:
            raise ValueError("LangchainEmbedding requires a model_name")
    elif name == MockEmbedding.__name__:
        return MockEmbedding.from_dict(data)
    else:
        raise ValueError(f"Invalid Embedding name: {name}")
