"""Init file."""

from llama_index.embeddings.google import GoogleUnivSentEncoderEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

__all__ = [
    "GoogleUnivSentEncoderEmbedding",
    "LangchainEmbedding",
    "OpenAIEmbedding",
]
