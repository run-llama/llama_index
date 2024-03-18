from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.together import TogetherEmbedding


def test_embedding_class():
    emb = TogetherEmbedding(model_name="fake_model_name", api_key="fake key")
    assert isinstance(emb, BaseEmbedding)
