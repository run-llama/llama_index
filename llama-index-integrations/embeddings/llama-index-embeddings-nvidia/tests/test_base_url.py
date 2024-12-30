import pytest
from llama_index.embeddings.nvidia import NVIDIAEmbedding as Interface
from pytest_httpx import HTTPXMock


@pytest.fixture()
def mock_local_models(httpx_mock: HTTPXMock, base_url: str):
    mock_response = {
        "data": [
            {
                "id": "model1",
                "object": "model",
                "created": 1234567890,
                "owned_by": "OWNER",
                "root": "model1",
            }
        ]
    }

    httpx_mock.add_response(
        url=f"{base_url}/models",
        method="GET",
        json=mock_response,
        status_code=200,
    )


# test case for base_url warning
@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8888/embeddings",
    ],
)
def test_base_url_invalid_not_hosted(base_url: str, mock_local_models) -> None:
    with pytest.warns(UserWarning) as msg:
        cls = Interface(base_url=base_url)
    assert cls._is_hosted is False
    assert len(msg) == 2
    assert "Expected format is " in str(msg[0].message)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8080/v1",
    ],
)
def test_base_url_valid_not_hosted(base_url: str, mock_local_models: None) -> None:
    with pytest.warns(UserWarning):
        cls = Interface(base_url=base_url)
    assert cls._is_hosted is False
    assert cls.model == "model1"


# @pytest.mark.parametrize("base_url", ["https://integrate.api.nvidia.com/v1/"])
# def test_base_url_valid_hosted(base_url: str) -> None:
#     Interface(base_url=base_url)
