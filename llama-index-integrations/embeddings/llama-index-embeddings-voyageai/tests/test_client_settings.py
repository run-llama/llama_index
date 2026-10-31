import pytest

from llama_index.embeddings.voyageai import VoyageEmbedding


@pytest.mark.parametrize(
    ("kwargs", "max_retries", "timeout"),
    [
        ({}, 0, None),
        ({"max_retries": 3, "timeout": 30.5}, 3, 30.5),
        ({"max_retries": 3}, 3, None),
        ({"timeout": 30.5}, 0, 30.5),
        ({"max_retries": 0, "timeout": None}, 0, None),
    ],
)
def test_client_settings(kwargs, max_retries, timeout):
    embedding = VoyageEmbedding(
        model_name="voyage-3.5", voyage_api_key="NOT_A_VALID_KEY", **kwargs
    )

    for client in (embedding._client, embedding._aclient):
        assert client.max_retries == max_retries
        assert client._params["request_timeout"] == timeout

    assert embedding.max_retries == max_retries
    assert embedding.timeout == timeout


def test_client_settings_round_trip():
    embedding = VoyageEmbedding(
        model_name="voyage-3.5",
        voyage_api_key="NOT_A_VALID_KEY",
        max_retries=3,
        timeout=30.5,
    )

    restored = VoyageEmbedding.from_dict(
        embedding.to_dict(), voyage_api_key="NOT_A_VALID_KEY"
    )

    for client in (restored._client, restored._aclient):
        assert client.max_retries == 3
        assert client._params["request_timeout"] == 30.5
