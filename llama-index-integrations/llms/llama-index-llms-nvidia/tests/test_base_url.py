import pytest
from llama_index.llms.nvidia import NVIDIA as Interface

from pytest_httpx import HTTPXMock


@pytest.fixture()
def mock_local_models(httpx_mock: HTTPXMock, base_url: str) -> None:
    mock_response = {
        "data": [
            {
                "id": "dummy",
                "object": "model",
                "created": 1234567890,
                "owned_by": "OWNER",
                "root": "model1",
            }
        ]
    }
    base_url = base_url.rstrip("/")
    httpx_mock.add_response(
        url=f"{base_url}/models",
        method="GET",
        json=mock_response,
        status_code=200,
    )


# test case for invalid base_url
@pytest.mark.parametrize(
    "base_url",
    [
        "http://0.0.0.0:8888/chat/completion",
        "https://0.0.0.0:8888/ranking",
    ],
)
def test_base_url_invalid_not_hosted(base_url: str, mock_local_models: None) -> None:
    Interface(base_url=base_url)


@pytest.mark.parametrize("base_url", ["http://localhost:8080/v1"])
def test_base_url_valid_not_hosted(base_url: str, mock_local_models: None) -> None:
    with pytest.warns(UserWarning):
        Interface(base_url=base_url)


@pytest.mark.parametrize("base_url", ["https://integrate.api.nvidia.com/v1/"])
def test_base_url_valid_hosted_without_api_key(base_url: str) -> None:
    Interface(base_url=base_url, api_key="BOGUS")


@pytest.mark.integration()
@pytest.mark.parametrize("base_url", ["https://integrate.api.nvidia.com/v1/"])
def test_base_url_valid_hosted_with_api_key(base_url: str) -> None:
    llm = Interface()
    assert llm.base_url == base_url

    llm = Interface(base_url=base_url)
    assert llm.base_url == base_url
