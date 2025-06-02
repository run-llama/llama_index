import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.upstage import UpstageEmbedding

UPSTAGE_TEST_API_KEY = "upstage_test_key"


@pytest.fixture()
def upstage_embedding():
    return pytest.importorskip(
        "llama_index.embeddings.upstage", reason="Cannot import UpstageEmbedding"
    ).UpstageEmbedding


@pytest.fixture()
def setup_environment(monkeypatch):
    monkeypatch.setenv("UPSTAGE_API_KEY", UPSTAGE_TEST_API_KEY)


def test_upstage_embedding_class(upstage_embedding):
    names_of_base_classes = [b.__name__ for b in upstage_embedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_upstage_embedding_fail_wrong_model(upstage_embedding):
    with pytest.raises(ValueError):
        upstage_embedding(model="foo")


def test_upstage_embedding_model_name(upstage_embedding):
    embedding = upstage_embedding(model="embedding")
    assert embedding._query_engine == "embedding-query"

    embedding = UpstageEmbedding(model="solar-embedding-1-large")
    assert embedding._query_engine == "solar-embedding-1-large-query"


def test_upstage_embedding_api_key_alias(upstage_embedding):
    embedding1 = upstage_embedding(api_key=UPSTAGE_TEST_API_KEY)
    embedding2 = upstage_embedding(upstage_api_key=UPSTAGE_TEST_API_KEY)
    embedding3 = upstage_embedding(error_api_key=UPSTAGE_TEST_API_KEY)

    assert embedding1.api_key == UPSTAGE_TEST_API_KEY
    assert embedding2.api_key == UPSTAGE_TEST_API_KEY
    assert embedding3.api_key == ""


def test_upstage_embedding_api_key_with_env(setup_environment, upstage_embedding):
    embedding = upstage_embedding()
    assert embedding.api_key == UPSTAGE_TEST_API_KEY
