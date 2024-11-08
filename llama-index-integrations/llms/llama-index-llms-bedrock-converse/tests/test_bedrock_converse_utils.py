import pytest
from llama_index.llms.bedrock_converse.utils import get_model_name


def test_get_model_name_translates_us():
    assert (
        get_model_name("us.meta.llama3-2-3b-instruct-v1:0")
        == "meta.llama3-2-3b-instruct-v1:0"
    )


def test_get_model_name_does_nottranslate_cn():
    assert (
        get_model_name("cn.meta.llama3-2-3b-instruct-v1:0")
        == "cn.meta.llama3-2-3b-instruct-v1:0"
    )


def test_get_model_name_does_nottranslate_unsupported():
    assert get_model_name("cohere.command-r-plus-v1:0") == "cohere.command-r-plus-v1:0"


def test_get_model_name_throws_inference_profile_exception():
    with pytest.raises(ValueError):
        assert get_model_name("us.cohere.command-r-plus-v1:0")
