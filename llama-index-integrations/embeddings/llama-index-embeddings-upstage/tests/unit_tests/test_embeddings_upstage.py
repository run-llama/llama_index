import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding


@pytest.fixture()
def upstage_embedding():
    return pytest.importorskip(
        "llama_index.embeddings.upstage", reason="Cannot import UpstageEmbedding"
    ).UpstageEmbedding


def test_upstage_embedding_class(upstage_embedding):
    names_of_base_classes = [b.__name__ for b in upstage_embedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_upstage_embedding_fail_wrong_model(upstage_embedding):
    with pytest.raises(ValueError):
        upstage_embedding(model="foo")


def test_upstage_embedding_api_key_alias(upstage_embedding):
    api_key = "test_key"
    embedding1 = upstage_embedding(api_key=api_key)
    embedding2 = upstage_embedding(upstage_api_key=api_key)
    embedding3 = upstage_embedding(error_api_key=api_key)

    assert embedding1.api_key == api_key
    assert embedding2.api_key == api_key
    assert embedding3.api_key == ""
