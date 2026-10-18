from unittest.mock import patch

import pytest
from llama_index.embeddings.alephalpha import AlephAlphaEmbedding

BASE = "llama_index.embeddings.alephalpha.base"


def _expected(**overrides):
    kwargs = {
        "host": "https://api.aleph-alpha.com",
        "hosting": "eu",
        "request_timeout_seconds": 5.0,
        "total_retries": 2,
        "nice": True,
        "verify_ssl": False,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture()
def embedding() -> AlephAlphaEmbedding:
    return AlephAlphaEmbedding(
        token="secret",
        hosting="eu",
        timeout=5.0,
        max_retries=2,
        nice=True,
        verify_ssl=False,
    )


def test_constructor_arguments_are_stored(embedding: AlephAlphaEmbedding) -> None:
    assert embedding.hosting == "eu"
    assert embedding.timeout == 5.0
    assert embedding.max_retries == 2
    assert embedding.nice is True
    assert embedding.verify_ssl is False


def test_sync_client_receives_settings(embedding: AlephAlphaEmbedding) -> None:
    with patch(f"{BASE}.Client") as client:
        embedding._get_client()
    assert client.call_args.kwargs == _expected(token="secret")


def test_async_client_receives_settings(embedding: AlephAlphaEmbedding) -> None:
    with patch(f"{BASE}.AsyncClient") as client:
        embedding._get_aclient()
    assert client.call_args.kwargs == _expected(token="secret")


def test_defaults_are_unchanged() -> None:
    embedding = AlephAlphaEmbedding(token="secret")
    assert embedding.hosting is None
    assert embedding.timeout is None
    assert embedding.max_retries == 10
    assert embedding.verify_ssl is True


@pytest.mark.parametrize(
    "kwargs", [{"timeout": -1.0}, {"max_retries": -1}], ids=["timeout", "max_retries"]
)
def test_invalid_values_are_rejected(kwargs: dict) -> None:
    # these fields are validated (ge=0), which only happens once they are forwarded
    with pytest.raises(ValueError):
        AlephAlphaEmbedding(token="secret", **kwargs)
