import pytest
from llama_index.embeddings.nvidia import NVIDIAEmbedding as Interface
from pytest_httpx import HTTPXMock
from requests_mock import Mocker
from contextlib import contextmanager
import os
from typing import Generator, Any


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


def test_create_without_base_url(public_class: type, monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "valid_api_key")
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    x = public_class()
    assert x.base_url == "https://integrate.api.nvidia.com/v1"
    assert str(x._client.base_url) == "https://integrate.api.nvidia.com/v1/"


# https.Url


def test_base_url_priority(public_class: type, monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "valid_api_key")
    ENV_URL = "https://ENV/v1"
    NV_PARAM_URL = "https://NV_PARAM/v1"
    PARAM_URL = "https://PARAM/v1"

    def get_base_url(**kwargs: Any) -> str:
        return public_class(model="NV-Embed-QA", **kwargs).base_url

    with no_env_var("NVIDIA_BASE_URL"):
        os.environ["NVIDIA_BASE_URL"] = ENV_URL
        assert get_base_url() == ENV_URL
        assert get_base_url(base_url=NV_PARAM_URL) == NV_PARAM_URL
        assert get_base_url(base_url=PARAM_URL) == PARAM_URL


@pytest.mark.parametrize(
    "base_url",
    [
        "bogus",
        "http:/",
        "http://",
        "http:/oops",
    ],
)
def test_param_base_url_negative(
    public_class: type, base_url: str, monkeypatch
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "valid_api_key")
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    with pytest.raises(ValueError) as e:
        public_class(model="model1", base_url=base_url)
    assert "Invalid base_url" in str(e.value)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8888/embeddings",
        "http://0.0.0.0:8888/rankings",
        "http://localhost:8888/embeddings/",
        "http://0.0.0.0:8888/rankings/",
        "http://localhost:8888/chat/completions",
        "http://localhost:8080/v1/embeddings",
        "http://0.0.0.0:8888/v1/rankings",
    ],
)
def test_expect_warn(public_class: type, base_url: str) -> None:
    with pytest.warns(UserWarning) as record:
        public_class(model="model1", base_url=base_url)
    assert len(record) == 1
    assert "does not end in /v1" in str(record[0].message)


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


@contextmanager
def no_env_var(var: str) -> Generator[None, None, None]:
    try:
        if val := os.environ.get(var, None):
            del os.environ[var]
        yield
    finally:
        if val:
            os.environ[var] = val
        else:
            if var in os.environ:
                del os.environ[var]


@pytest.mark.parametrize(
    "base_url",
    [
        "http://host/path0/path1/path2/v1",
        "http://host:123/path0/path1/path2/v1/",
    ],
)
def test_proxy_base_url(
    public_class: type, base_url: str, requests_mock: Mocker
) -> None:
    with no_env_var("NVIDIA_BASE_URL"):
        client = public_class(
            api_key="NO_API_KEY_PROVIDED", model="NV-Embed-QA", base_url=base_url
        )
        assert base_url.startswith(client.base_url)
