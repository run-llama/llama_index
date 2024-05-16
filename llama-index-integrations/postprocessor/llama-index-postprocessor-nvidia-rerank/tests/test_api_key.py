import os

import pytest

from llama_index.postprocessor.nvidia_rerank import NVIDIARerank
from llama_index.core.schema import NodeWithScore, Document

from typing import Any
from .conftest import no_env_var


def get_api_key(instance: Any) -> str:
    return instance._api_key


def test_create_without_api_key() -> None:
    with no_env_var("NVIDIA_API_KEY"):
        NVIDIARerank()


@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_create_with_api_key(param: str) -> None:
    with no_env_var("NVIDIA_API_KEY"):
        instance = NVIDIARerank(**{param: "just testing no failure"})
        assert get_api_key(instance) == "just testing no failure"


def test_api_key_priority() -> None:
    with no_env_var("NVIDIA_API_KEY"):
        os.environ["NVIDIA_API_KEY"] = "ENV"
        assert get_api_key(NVIDIARerank()) == "ENV"
        assert get_api_key(NVIDIARerank(nvidia_api_key="PARAM")) == "PARAM"
        assert get_api_key(NVIDIARerank(api_key="PARAM")) == "PARAM"
        assert get_api_key(NVIDIARerank(api_key="LOW", nvidia_api_key="HIGH")) == "HIGH"


@pytest.mark.integration()
def test_missing_api_key_error() -> None:
    with no_env_var("NVIDIA_API_KEY"):
        client = NVIDIARerank()
        with pytest.raises(Exception) as exc_info:
            client.postprocess_nodes(
                [NodeWithScore(node=Document(text="Hello, world!"))],
                query_str="Hello, world!",
            )
        message = str(exc_info.value)
        assert "401" in message


@pytest.mark.integration()
def test_bogus_api_key_error() -> None:
    with no_env_var("NVIDIA_API_KEY"):
        client = NVIDIARerank(nvidia_api_key="BOGUS")
        with pytest.raises(Exception) as exc_info:
            client.postprocess_nodes(
                [NodeWithScore(node=Document(text="Hello, world!"))],
                query_str="Hello, world!",
            )
        message = str(exc_info.value)
        assert "401" in message


@pytest.mark.integration()
@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_api_key(param: str, model: str, mode: dict) -> None:
    api_key = os.environ.get("NVIDIA_API_KEY")
    with no_env_var("NVIDIA_API_KEY"):
        client = NVIDIARerank(**{"model": model, param: api_key}).mode(**mode)
        assert client.postprocess_nodes(
            [NodeWithScore(node=Document(text="Hello, world!"))],
            query_str="Hello, world!",
        )
