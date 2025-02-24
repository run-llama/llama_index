import os

import pytest
import respx
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank as Interface
from llama_index.core.schema import NodeWithScore, Document

from typing import Any


@pytest.fixture()
def mock_local_models(respx_mock: respx.MockRouter) -> None:
    respx_mock.get("https://test_url/v1/models").respond(
        json={"data": [{"id": "model1"}]}
    )


def get_api_key(instance: Any) -> str:
    return instance._api_key


def test_create_default_url_without_api_key(masked_env_var: str) -> None:
    with pytest.raises(ValueError) as e:
        Interface()
    assert "API key is required" in str(e.value)


@pytest.mark.usefixtures("mock_local_models")
def test_create_unknown_url_without_api_key(masked_env_var: str) -> None:
    Interface(base_url="https://test_url/v1")


@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_create_with_api_key(param: str, masked_env_var: str) -> None:
    instance = Interface(**{param: "just testing no failure"})
    assert get_api_key(instance) == "just testing no failure"


def test_api_key_priority(masked_env_var: str) -> None:
    try:
        os.environ["NVIDIA_API_KEY"] = "ENV"
        assert get_api_key(Interface()) == "ENV"
        assert get_api_key(Interface(nvidia_api_key="PARAM")) == "PARAM"
        assert get_api_key(Interface(api_key="PARAM")) == "PARAM"
        assert get_api_key(Interface(api_key="LOW", nvidia_api_key="HIGH")) == "HIGH"
    finally:
        # we must clean up environ or it may impact other tests
        del os.environ["NVIDIA_API_KEY"]


@pytest.mark.integration()
def test_bogus_api_key_error(masked_env_var: str) -> None:
    client = Interface(nvidia_api_key="BOGUS")
    with pytest.raises(Exception) as exc_info:
        client.postprocess_nodes(
            [NodeWithScore(node=Document(text="Hello, world!"))],
            query_str="Hello, world!",
        )
    message = str(exc_info.value)
    assert "401" in message


@pytest.mark.integration()
@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_api_key(model: str, mode: dict, param: str, masked_env_var: str) -> None:
    client = Interface(model=model, **{**mode, **{param: masked_env_var}})
    assert client.postprocess_nodes(
        [NodeWithScore(node=Document(text="Hello, world!"))],
        query_str="Hello, world!",
    )
