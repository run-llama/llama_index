from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.paieas import PaiEas
import os
from urllib.parse import urljoin


def test_text_inference_embedding_class():
    names_of_base_classes = [b.__name__ for b in PaiEas.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def _get_eas_llm() -> PaiEas:
    eas_url = os.environ.get("TEST_EAS_URL", None)
    eas_token = os.environ.get("TEST_EAS_TOKEN", None)

    if not eas_url or not eas_token:
        return None

    return PaiEas(api_key=eas_token, api_base=urljoin(eas_url, "v1"))


def test_pai_eas_llm_complete():
    eas_llm = _get_eas_llm()
    if eas_llm:
        response = eas_llm.complete("What is Alibaba Cloud?")
        print(response.text)
        assert len(response.text) > 0


def test_pai_eas_llm_stream_complete():
    eas_llm = _get_eas_llm()

    if eas_llm:
        response = eas_llm.stream_complete("What is Alibaba Cloud?")
        text = None
        for r in response:
            print(r.text)
            text = r.text

        assert len(text) > 0


def test_pai_eas_llm_chat():
    eas_llm = _get_eas_llm()
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


def test_pai_eas_llm_stream_chat():
    eas_llm = _get_eas_llm()
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
