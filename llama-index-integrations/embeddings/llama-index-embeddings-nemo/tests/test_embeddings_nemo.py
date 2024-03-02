from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.nemo import NeMoEmbedding


def test_embedding_class():
    emb = NeMoEmbedding()
    assert isinstance(emb, BaseEmbedding)
