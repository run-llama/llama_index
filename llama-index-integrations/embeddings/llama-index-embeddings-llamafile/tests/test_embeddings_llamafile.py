from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.llamafile import LlamafileEmbedding


def test_embedding_class():
    emb = LlamafileEmbedding()
    assert isinstance(emb, BaseEmbedding)
