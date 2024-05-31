from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.embeddings.jinaai import JinaEmbedding, JinaMultiModalEmbedding


def test_embedding_class():
    emb = JinaEmbedding()
    assert isinstance(emb, BaseEmbedding)

    emb = JinaMultiModalEmbedding()
    assert isinstance(emb, MultiModalEmbedding)
