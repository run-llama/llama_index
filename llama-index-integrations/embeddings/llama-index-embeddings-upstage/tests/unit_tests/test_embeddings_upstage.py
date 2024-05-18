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


def upstage_embedding_fail_wrong_model(upstage_embedding):
    with pytest.raises(ValueError):
        upstage_embedding(model="foo")
