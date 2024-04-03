from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.nvidia import NVIDIAEmbedding


def test_embedding_class():
    emb = NVIDIAEmbedding()
    assert isinstance(emb, BaseEmbedding)
