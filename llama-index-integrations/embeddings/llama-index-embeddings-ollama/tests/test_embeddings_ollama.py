from llama_index.core.embeddings.base import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding


def test_embedding_class():
    emb = OllamaEmbedding(model_name="")
    assert isinstance(emb, BaseEmbedding)
