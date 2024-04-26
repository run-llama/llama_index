import pytest

from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

from .conftest import no_env_var

# we don't test this because we do not want to force users to have an API key
#  NVIDIARerank().mode("nim", base_url=...) must work without an API key
# def test_mode_switch_nvidia_throws_without_key():
#     emb = NVIDIARerank()
#     with pytest.raises(ValueError):
#         emb.mode("nvidia")


def test_mode_switch_nvidia_with_key():
    with no_env_var("NVIDIA_API_KEY"):
        NVIDIARerank().mode("nvidia", api_key="test")


def test_mode_switch_nim_throws_without_url():
    instance = NVIDIARerank()
    with pytest.raises(ValueError):
        instance.mode("nim")


def test_mode_switch_nim_with_url():
    NVIDIARerank().mode("nim", base_url="test")


def test_mode_switch_param_setting():
    instance0 = NVIDIARerank(model="dummy")

    isntance1 = instance0.mode("nim", base_url="https://test_url/v1/")
    assert isntance1.model == "dummy"
    assert str(isntance1._client.base_url) == "https://test_url/v1/"
    assert str(isntance1._aclient.base_url) == "https://test_url/v1/"

    instance2 = isntance1.mode("nvidia", api_key="test", model="dummy-2")
    assert instance2.model == "dummy-2"
    assert (
        str(instance2._client.base_url)
        == "https://ai.api.nvidia.com/v1/retrieval/nvidia/"
    )
    assert (
        str(instance2._aclient.base_url)
        == "https://ai.api.nvidia.com/v1/retrieval/nvidia/"
    )
    assert instance2._client.api_key == "test"
    assert instance2._aclient.api_key == "test"
