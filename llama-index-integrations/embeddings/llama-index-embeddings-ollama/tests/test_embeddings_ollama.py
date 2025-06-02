from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding


def test_embedding_class():
    emb = OllamaEmbedding(
        model_name="", client_kwargs={"headers": {"Authorization": "Bearer token"}}
    )
    assert isinstance(emb, BaseEmbedding)
