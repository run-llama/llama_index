from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel


def test_deepinfra_embedding_class():
    model = DeepInfraEmbeddingModel()
    assert isinstance(model, BaseEmbedding)
