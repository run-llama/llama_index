from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.solar import SolarEmbedding


def test_embedding_class():
    emb = SolarEmbedding()
    assert isinstance(emb, BaseEmbedding)
