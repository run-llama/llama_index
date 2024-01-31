from llama_index.core.embeddings.base import BaseEmbedding
from llama_index.embeddings.jinaai import JinaEmbedding


def test_embedding_class():
    emb = JinaEmbedding()
    assert isinstance(emb, BaseEmbedding)
