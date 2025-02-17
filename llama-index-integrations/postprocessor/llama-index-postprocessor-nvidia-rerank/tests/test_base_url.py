from urllib.parse import urlparse, urlunparse

import pytest
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank as Interface
from llama_index.postprocessor.nvidia_rerank.utils import BASE_URL
import respx


@pytest.fixture()
def mock_v1_local_models2(respx_mock: respx.MockRouter, base_url: str) -> None:
    parsed = urlparse(base_url)
    normalized_path = parsed.path.rstrip("/")
    if not normalized_path.endswith("/v1"):
        normalized_path += "/v1"
        base_url = urlunparse(
            (parsed.scheme, parsed.netloc, normalized_path, None, None, None)
        )
    # Intercept GET call for retrieving models using httpx.
    respx_mock.get(f"{base_url}/models").respond(
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
        }
    )


# Updated test for non-hosted URLs that may need normalization.
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
    parsed = urlparse(base_url)
    normalized_path = parsed.path.rstrip("/")
    # Expect a warning if the URL does NOT already end with "/v1"
    if not normalized_path.endswith("/v1"):
        with pytest.warns(UserWarning, match="does not end in /v1"):
            client = Interface(base_url=base_url)
    else:
        client = Interface(base_url=base_url)
    # Assert that the client's base_url is normalized to end with '/v1'
    assert client.base_url.endswith("/v1")


# Updated test for valid non-hosted URL.
@pytest.mark.parametrize(
    "base_url",
    [
        "http://0.0.0.0:8888/v1",
    ],
)
def test_base_url_valid_not_hosted(base_url: str, mock_v1_local_models2: None) -> None:
    # The default model warning is expected in non-hosted mode
    with pytest.warns(UserWarning, match="Default model is set") as record:
        client = Interface(base_url=base_url)
    # Also verify the base_url remains normalized (unchanged in this case)
    assert client.base_url.endswith("/v1")


# Updated test for hosted base URL.
@pytest.mark.parametrize(
    "base_url",
    [BASE_URL],
)
def test_base_url_valid_hosted(base_url: str, mock_v1_local_models2: None) -> None:
    client = Interface(base_url=base_url, api_key="BOGUS")
    assert client._is_hosted
    # Hosted client should use the provided base_url exactly.
    assert client.base_url == base_url


# Updated test for proxy base URLs.
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
    # Since the URL is already normalized, verify it remains unchanged.
    assert client.base_url == base_url


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
