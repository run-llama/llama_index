from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.fireworks import FireworksEmbedding


def test_fireworks_class():
    emb = FireworksEmbedding()
    assert isinstance(emb, BaseEmbedding)
