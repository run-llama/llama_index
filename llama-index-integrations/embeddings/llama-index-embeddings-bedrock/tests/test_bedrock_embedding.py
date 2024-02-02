from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.bedrock import BedrockEmbedding


def test_anyscale_class():
    emb = BedrockEmbedding()
    assert isinstance(emb, BaseEmbedding)
