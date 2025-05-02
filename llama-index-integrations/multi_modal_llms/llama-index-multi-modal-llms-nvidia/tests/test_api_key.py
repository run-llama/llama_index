import os

import pytest

from llama_index.multi_modal_llms.nvidia import NVIDIAMultiModal

from typing import Any
from llama_index.core.schema import ImageDocument


def get_api_key(instance: Any) -> str:
    return instance.api_key


def test_create_default_url_without_api_key(masked_env_var: str) -> None:
    with pytest.raises(ValueError) as err_msg:
        NVIDIAMultiModal()
    assert (
        str(err_msg.value)
        == "An API key is required for the hosted NIM. This will become an error in 0.2.0."
    )


@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_create_with_api_key(param: str, masked_env_var: str) -> None:
    instance = NVIDIAMultiModal(**{param: "just testing no failure"})
    assert get_api_key(instance) == "just testing no failure"


def test_api_key_priority(masked_env_var: str) -> None:
    try:
        os.environ["NVIDIA_API_KEY"] = "ENV"
        assert get_api_key(NVIDIAMultiModal()) == "ENV"
        assert get_api_key(NVIDIAMultiModal(nvidia_api_key="PARAM")) == "PARAM"
        assert get_api_key(NVIDIAMultiModal(api_key="PARAM")) == "PARAM"
        assert (
            get_api_key(NVIDIAMultiModal(api_key="LOW", nvidia_api_key="HIGH"))
            == "HIGH"
        )
    finally:
        # we must clean up environ or it may impact other tests
        del os.environ["NVIDIA_API_KEY"]


@pytest.mark.integration()
def test_bogus_api_key_error(vlm_model: str, masked_env_var: str) -> None:
    client = NVIDIAMultiModal(model=vlm_model, nvidia_api_key="BOGUS")
    with pytest.raises(Exception) as exc_info:
        client.complete(
            prompt="xyz", image_documents=[ImageDocument(image_url="https://xyz.com")]
        )
    assert "401" in str(exc_info.value)
