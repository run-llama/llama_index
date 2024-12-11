import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.upstage import Upstage


def test_text_inference_llm_class():
    names_of_base_classes = [b.__name__ for b in Upstage.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_upstage_llm_api_key_alias():
    api_key = "test_key"
    llm1 = Upstage(api_key=api_key)
    llm2 = Upstage(upstage_api_key=api_key)
    llm3 = Upstage(error_api_key=api_key)

    assert llm1.api_key == api_key
    assert llm2.api_key == api_key
    assert llm3.api_key == ""


def test_upstage_tokenizer():
    llm = Upstage()
    tokenizer = llm._tokenizer

    with pytest.raises(Exception):
        llm = Upstage(tokenizer_name="wrong name")
        tokenizer = llm._tokenizer


def test_upstage_tokenizer_count_tokens():
    llm = Upstage()
    assert (
        llm.get_num_tokens_from_message(
            [ChatMessage(role=MessageRole.USER, content="Hello World")]
        )
        == 12
    )
