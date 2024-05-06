from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel


def test_deepinfra_embedding_class():
    model = DeepInfraEmbeddingModel()
    assert isinstance(model, BaseEmbedding)


def test_deepinfra_query_prefix():
    model = DeepInfraEmbeddingModel(query_prefix="query")
    result = model._add_query_prefix(["test"])
    assert result == ["querytest"]


def test_deepinfra_text_prefix():
    model = DeepInfraEmbeddingModel(text_prefix="text")
    result = model._add_text_prefix(["test"])
    assert result == ["texttest"]


def test_deepinfra_default_query_prefix():
    model = DeepInfraEmbeddingModel()
    result = model._add_query_prefix(["test"])
    assert result == ["test"]


def test_deepinfra_default_text_prefix():
    model = DeepInfraEmbeddingModel()
    result = model._add_text_prefix(["test"])
    assert result == ["test"]
