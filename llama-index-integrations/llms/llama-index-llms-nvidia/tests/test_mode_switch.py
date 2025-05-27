import pytest

from llama_index.llms.nvidia import NVIDIA as Interface
from llama_index.llms.nvidia.base import BASE_URL
from pytest_httpx import HTTPXMock

UNKNOWN_URLS = [
    "https://test_url/v1",
    "https://test_url/v1/",
    "http://test_url/v1",
    "http://test_url/v1/",
]


@pytest.fixture()
def mock_unknown_urls(httpx_mock: HTTPXMock, base_url: str) -> None:
    mock_response = {
        "data": [
            {
                "id": "meta/llama3-8b-instruct",
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


def test_mode_switch_nvidia_throws_without_key_deprecated(masked_env_var: str):
    x = Interface()
    with pytest.raises(ValueError):
        with pytest.warns(DeprecationWarning):
            x.mode("nvidia")


def test_mode_switch_nvidia_with_key_deprecated(masked_env_var: str):
    with pytest.warns(DeprecationWarning):
        Interface().mode("nvidia", api_key="test")


def test_mode_switch_nim_throws_without_url_deprecated():
    instance = Interface()
    with pytest.raises(ValueError):
        with pytest.warns(DeprecationWarning):
            instance.mode("nim")


def test_mode_switch_nim_with_url_deprecated():
    with pytest.warns(DeprecationWarning):
        Interface().mode("nim", base_url="test")


@pytest.mark.parametrize("base_url", ["https://test_url/v1/"])
def test_mode_switch_param_setting_deprecated(base_url):
    instance = Interface(model="meta/llama3-8b-instruct")

    with pytest.warns(DeprecationWarning):
        instance1 = instance.mode("nim", base_url=base_url)
    assert instance1.model == "meta/llama3-8b-instruct"
    assert str(instance1.api_base) == base_url

    with pytest.warns(DeprecationWarning):
        instance2 = instance1.mode(
            "nvidia", api_key="test", model="meta/llama3-15b-instruct"
        )
    assert instance2.model == "meta/llama3-15b-instruct"
    assert str(instance2.api_base) == BASE_URL
    assert instance2.api_key == "test"


@pytest.mark.parametrize("base_url", UNKNOWN_URLS)
def test_mode_switch_unknown_base_url_without_key(
    mock_unknown_urls, masked_env_var: str, base_url: str
):
    Interface(base_url=base_url)


@pytest.mark.parametrize("base_url", UNKNOWN_URLS)
@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_mode_switch_unknown_base_url_with_key(
    mock_unknown_urls, masked_env_var: str, param: str, base_url: str
):
    Interface(base_url=base_url, **{param: "test"})


@pytest.mark.parametrize("base_url", [BASE_URL])
def test_mode_switch_known_base_url_without_key(masked_env_var: str, base_url: str):
    with pytest.warns(UserWarning):
        cls = Interface(base_url=base_url)
        assert cls._is_hosted


@pytest.mark.parametrize("base_url", [BASE_URL])
@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_mode_switch_known_base_url_with_key(
    masked_env_var: str, base_url: str, param: str
):
    Interface(base_url=base_url, **{param: "test"})
