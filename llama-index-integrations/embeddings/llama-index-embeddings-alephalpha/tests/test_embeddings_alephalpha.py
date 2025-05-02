from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.alephalpha import AlephAlphaEmbedding


def test_anyscale_class():
    emb = AlephAlphaEmbedding(token="fake_token", model="luminous-base")
    assert isinstance(emb, BaseEmbedding)
