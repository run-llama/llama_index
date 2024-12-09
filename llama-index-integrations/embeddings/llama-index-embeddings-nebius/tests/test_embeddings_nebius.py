from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.nebius import NebiusEmbedding


def test_embedding_class():
    emb = NebiusEmbedding(model_name="per_aspera", api_key="ad_astra")
    assert isinstance(emb, BaseEmbedding)
