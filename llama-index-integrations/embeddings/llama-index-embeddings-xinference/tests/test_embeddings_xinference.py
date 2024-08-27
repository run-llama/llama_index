from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.xinference import XinferenceEmbedding


def test_embedding_class():
    emb = XinferenceEmbedding(model_uid="")
    assert isinstance(emb, BaseEmbedding)
