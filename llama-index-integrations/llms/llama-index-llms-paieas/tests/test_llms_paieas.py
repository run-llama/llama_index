import os
import pytest

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.paieas import PaiEas

should_skip = os.environ.get("PAIEAS_API_KEY", None) is None


def test_text_inference_embedding_class():
    names_of_base_classes = [b.__name__ for b in PaiEas.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@pytest.mark.skipif(should_skip, reason="PAIEAS_API_KEY is not set.")
def test_pai_eas_llm_complete():
    eas_llm = PaiEas()
    if eas_llm:
        response = eas_llm.complete("What is Alibaba Cloud?")
        print(response.text)
        assert len(response.text) > 0


@pytest.mark.skipif(should_skip, reason="PAIEAS_API_KEY is not set.")
def test_pai_eas_llm_stream_complete():
    eas_llm = PaiEas()

    if eas_llm:
        response = eas_llm.stream_complete("What is Alibaba Cloud?")
        text = None
        for r in response:
            print(r.text)
            text = r.text

        assert len(text) > 0


@pytest.mark.skipif(should_skip, reason="PAIEAS_API_KEY is not set.")
def test_pai_eas_llm_chat():
    eas_llm = PaiEas()
    if eas_llm:
        chat_messages = [
            ChatMessage(
                role="user",
                content="What is Alibaba Cloud?",
            )
        ]
        response = eas_llm.chat(chat_messages)
        print(response.message.content)
        assert len(response.message.content) > 0


@pytest.mark.skipif(should_skip, reason="PAIEAS_API_KEY is not set.")
def test_pai_eas_llm_stream_chat():
    eas_llm = PaiEas()
    if eas_llm:
        chat_messages = [
            ChatMessage(
                role="user",
                content="What is Alibaba Cloud?",
            )
        ]
        response = eas_llm.stream_chat(chat_messages)

        text = None
        for r in response:
            text = r.message.content
        print(text)
        assert len(text) > 0
