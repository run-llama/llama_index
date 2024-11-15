import pytest
from llama_index.llms.nvidia import NVIDIA as Interface
from pytest_httpx import HTTPXMock


@pytest.fixture()
def mock_local_models(httpx_mock: HTTPXMock, base_url: str):
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

    if base_url.endswith("/"):
        base_url = base_url[:-1]

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
        "localhost",
        "localhost:8888",
        "http://localhost:8888/embeddings",
        "http://0.0.0.0:8888/rankings",
        "http://localhost:8888/chat/completions",
        "http://test_url/.../v1",
        "https://test_url/.../v1",
    ],
)
def test_base_url_invalid_not_hosted(base_url: str) -> None:
    with pytest.raises(ValueError) as msg:
        Interface(base_url=base_url)
    assert "Invalid base_url" in str(msg.value)


@pytest.mark.parametrize("base_url", ["http://localhost:8080/v1/"])
def test_base_url_valid_not_hosted(base_url: str, mock_local_models: None) -> None:
    with pytest.warns(UserWarning):
        Interface(base_url=base_url)


@pytest.mark.parametrize("base_url", ["https://integrate.api.nvidia.com/v1/"])
def test_base_url_valid_hosted(base_url: str) -> None:
    Interface(base_url=base_url)
