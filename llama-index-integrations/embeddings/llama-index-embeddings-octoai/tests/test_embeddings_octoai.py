from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.octoai import OctoAIEmbedding


def test_fireworks_class():
    emb = OctoAIEmbedding()
    assert isinstance(emb, BaseEmbedding)
