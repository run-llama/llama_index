from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.mistralai import MistralAIEmbedding


def test_embedding_class():
    emb = MistralAIEmbedding(api_key="fake-key")
    assert isinstance(emb, BaseEmbedding)
