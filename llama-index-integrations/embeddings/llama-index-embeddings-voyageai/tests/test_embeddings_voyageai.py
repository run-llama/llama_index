from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding


def test_embedding_class():
    emb = VoyageEmbedding(model_name="")
    assert isinstance(emb, BaseEmbedding)
