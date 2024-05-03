from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel


def test_deepinfra_embedding_class():
    model = DeepInfraEmbeddingModel()
    assert isinstance(model, BaseEmbedding)


def test_deepinfra_embedding_request():
    model = DeepInfraEmbeddingModel()

    model._post = lambda data: {"embeddings": [[0.1, 0.2, 0.3]]}
    assert model.get_text_embedding("Hello, world!") == [0.1, 0.2, 0.3]


def test_deepinfra_embedding_request_list():
    model = DeepInfraEmbeddingModel()

    model._post = lambda data: {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
    assert model.get_text_embedding(["Hello, world!", "Goodbye, world!"]) == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
