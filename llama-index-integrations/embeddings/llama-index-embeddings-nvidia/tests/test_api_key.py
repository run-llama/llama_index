import os

import pytest

from llama_index.embeddings.nvidia import NVIDIAEmbedding

from typing import Any
from .conftest import no_env_var


def get_api_key(instance: Any) -> str:
    return instance._client.api_key


def test_create_without_api_key() -> None:
    with no_env_var("NVIDIA_API_KEY"):
        NVIDIAEmbedding()


@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_create_with_api_key(param: str) -> None:
    with no_env_var("NVIDIA_API_KEY"):
        instance = NVIDIAEmbedding(**{param: "just testing no failure"})
        assert get_api_key(instance) == "just testing no failure"


def test_api_key_priority() -> None:
    with no_env_var("NVIDIA_API_KEY"):
        os.environ["NVIDIA_API_KEY"] = "ENV"
        assert get_api_key(NVIDIAEmbedding()) == "ENV"
        assert get_api_key(NVIDIAEmbedding(nvidia_api_key="PARAM")) == "PARAM"
        assert get_api_key(NVIDIAEmbedding(api_key="PARAM")) == "PARAM"
        assert (
            get_api_key(NVIDIAEmbedding(api_key="LOW", nvidia_api_key="HIGH")) == "HIGH"
        )


@pytest.mark.integration()
def test_missing_api_key_error() -> None:
    with no_env_var("NVIDIA_API_KEY"):
        client = NVIDIAEmbedding()
        with pytest.raises(Exception) as exc_info:
            client.get_query_embedding("Hello, world!")
        message = str(exc_info.value)
        assert "401" in message


@pytest.mark.integration()
def test_bogus_api_key_error() -> None:
    with no_env_var("NVIDIA_API_KEY"):
        client = NVIDIAEmbedding(nvidia_api_key="BOGUS")
        with pytest.raises(Exception) as exc_info:
            client.get_query_embedding("Hello, world!")
        message = str(exc_info.value)
        assert "401" in message


@pytest.mark.integration()
@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_api_key(param: str, model: str, mode: dict) -> None:
    api_key = os.environ.get("NVIDIA_API_KEY")
    with no_env_var("NVIDIA_API_KEY"):
        client = NVIDIAEmbedding(**{"model": model, param: api_key}).mode(**mode)
        assert client.get_query_embedding("Hello, world!")
