from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.premai import PremAIEmbeddings


def test_embedding_class():
    emb = PremAIEmbeddings(api_key="fake-key")
    assert isinstance(emb, BaseEmbedding)
