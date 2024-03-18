from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.cohere import CohereEmbedding


def test_anyscale_class():
    emb = CohereEmbedding(cohere_api_key="fake_key")
    assert isinstance(emb, BaseEmbedding)
