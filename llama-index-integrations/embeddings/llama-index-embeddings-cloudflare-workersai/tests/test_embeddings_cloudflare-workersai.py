from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.cloudflare_workersai import CloudflareEmbedding


def test_embedding_class():
    emb = CloudflareEmbedding(account_id="fake_id")
    assert isinstance(emb, BaseEmbedding)
