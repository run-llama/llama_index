from urllib.parse import urlparse, urlunparse

import pytest
from requests_mock import Mocker
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank as Interface


@pytest.fixture()
def mock_v1_local_models2(requests_mock: Mocker, base_url: str) -> None:
    parsed = urlparse(base_url)
    normalized_path = parsed.path.rstrip("/")
    if not normalized_path.endswith("/v1"):
        normalized_path += "/v1"
        base_url = urlunparse(
            (parsed.scheme, parsed.netloc, normalized_path, None, None, None)
        )
    requests_mock.get(
        f"{base_url}/models",
        json={
            "data": [
                {
                    "id": "model1",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                    "root": "model1",
                },
            ]
        },
    )


# test case for invalid base_url
@pytest.mark.parametrize(
    "base_url",
    [
        "http://0.0.0.0:8888/rankings",
        "http://0.0.0.0:8888/ranking",
        "http://test_url/.../v1",
        "https://test_url/.../v1",
    ],
)
def test_base_url_invalid_not_hosted(
    base_url: str, mock_v1_local_models2: None
) -> None:
    Interface(base_url=base_url)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://0.0.0.0:8888/v1",
    ],
)
def test_base_url_valid_not_hosted(base_url: str, mock_v1_local_models2: None) -> None:
    with pytest.warns(UserWarning) as record:
        Interface(base_url=base_url)
    assert "Default model is set" in str(record[0].message)


@pytest.mark.parametrize(
    "base_url",
    ["https://ai.api.nvidia.com/v1"],
)
def test_base_url_valid_hosted(base_url: str, mock_v1_local_models2: None) -> None:
    Interface(base_url=base_url, api_key="BOGUS")


@pytest.mark.parametrize(
    "base_url",
    [
        "bogus",
        "http:/",
        "http://",
        "http:/oops",
    ],
)
def test_param_base_url_negative(base_url: str, monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "valid_api_key")
    with pytest.raises(ValueError) as e:
        Interface(model="model1", base_url=base_url)
    assert "Invalid base_url" in str(e.value)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://host/path0/path1/path2/v1",
        "http://host:123/path0/path1/path2/v1",
    ],
)
def test_proxy_base_url(base_url: str, mock_v1_local_models2: None) -> None:
    client = Interface(api_key="NO_API_KEY_PROVIDED", base_url=base_url)
    assert not client._is_hosted
    assert base_url.startswith(client.base_url)
