from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.netmind import NetmindEmbedding


def test_embedding_class():
    emb = NetmindEmbedding(model_name="fake_model_name", api_key="fake key")
    assert isinstance(emb, BaseEmbedding)
