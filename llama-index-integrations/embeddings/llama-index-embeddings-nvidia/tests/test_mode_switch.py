import pytest

from llama_index.embeddings.nvidia import NVIDIAEmbedding as Interface
from llama_index.embeddings.nvidia.base import BASE_URL, KNOWN_URLS


def test_mode_switch_throws_without_key_deprecated(masked_env_var: str):
    x = Interface()
    with pytest.raises(ValueError):
        with pytest.warns(DeprecationWarning):
            x.mode("nvidia")


def test_mode_switch_with_key_deprecated(masked_env_var: str):
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


def test_mode_switch_param_setting_deprecated():
    instance = Interface(model="dummy")

    with pytest.warns(DeprecationWarning):
        instance1 = instance.mode("nim", base_url="https://test_url/v1/")
    assert instance1.model == "dummy"
    assert str(instance1._client.base_url) == "https://test_url/v1/"

    with pytest.warns(DeprecationWarning):
        instance2 = instance1.mode("nvidia", api_key="test", model="dummy-2")
    assert instance2.model == "dummy-2"
    assert str(instance2._client.base_url) == BASE_URL
    assert instance2._client.api_key == "test"


UNKNOWN_URLS = [
    "https://test_url/v1",
    "https://test_url/v1/",
    "http://test_url/v1",
    "http://test_url/v1/",
]


@pytest.mark.parametrize("base_url", UNKNOWN_URLS)
def test_mode_switch_unknown_base_url_without_key(masked_env_var: str, base_url: str):
    Interface(base_url=base_url)


@pytest.mark.parametrize("base_url", UNKNOWN_URLS)
@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_mode_switch_unknown_base_url_with_key(
    masked_env_var: str, param: str, base_url: str
):
    Interface(base_url=base_url, **{param: "test"})


@pytest.mark.parametrize("base_url", KNOWN_URLS)
def test_mode_switch_known_base_url_without_key(masked_env_var: str, base_url: str):
    with pytest.warns(UserWarning):
        Interface(base_url=base_url)


@pytest.mark.parametrize("base_url", KNOWN_URLS)
@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_mode_switch_known_base_url_with_key(
    masked_env_var: str, base_url: str, param: str
):
    Interface(base_url=base_url, **{param: "test"})
