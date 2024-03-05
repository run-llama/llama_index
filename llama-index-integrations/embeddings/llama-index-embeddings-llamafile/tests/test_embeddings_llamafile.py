from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.llamafile import LlamafileEmbedding


def test_embedding_class():
    emb = LlamafileEmbedding()
    assert isinstance(emb, BaseEmbedding)


def test_init():
    emb = LlamafileEmbedding()
    assert emb.request_timeout == 30.0
    emb = LlamafileEmbedding(request_timeout=10.0)
    assert emb.request_timeout == 10.0
