import pytest

from llama_index.embeddings.nvidia import NVIDIAEmbedding

from .conftest import no_env_var

# we don't test this because we do not want to force users to have an API key
#  NVIDIAEmbedding().mode("nim", base_url=...) must work without an API key
# def test_mode_switch_nvidia_throws_without_key():
#     emb = NVIDIAEmbedding()
#     with pytest.raises(ValueError):
#         emb.mode("nvidia")


def test_mode_switch_nvidia_with_key():
    with no_env_var("NVIDIA_API_KEY"):
        NVIDIAEmbedding().mode("nvidia", api_key="test")


def test_mode_switch_nim_throws_without_url():
    emb = NVIDIAEmbedding()
    with pytest.raises(ValueError):
        emb.mode("nim")


def test_mode_switch_nim_with_url():
    NVIDIAEmbedding().mode("nim", base_url="test")


def test_mode_switch_param_setting():
    emb = NVIDIAEmbedding(model="dummy")

    nim_emb = emb.mode("nim", base_url="https://test_url/v1/")
    assert nim_emb.model == "dummy"
    assert str(nim_emb._client.base_url) == "https://test_url/v1/"
    assert str(nim_emb._aclient.base_url) == "https://test_url/v1/"

    cat_emb = nim_emb.mode("nvidia", api_key="test", model="dummy-2")
    assert cat_emb.model == "dummy-2"
    assert (
        str(cat_emb._client.base_url)
        == "https://ai.api.nvidia.com/v1/retrieval/nvidia/"
    )
    assert (
        str(cat_emb._aclient.base_url)
        == "https://ai.api.nvidia.com/v1/retrieval/nvidia/"
    )
    assert cat_emb._client.api_key == "test"
    assert cat_emb._aclient.api_key == "test"
