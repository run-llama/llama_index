import os
from typing import Any, AsyncGenerator, Generator

from llama_index.llms.base import ChatMessage
from llama_index.llms.konko import Konko



def test_chat_model_basic_non_openai_model() -> None:
        llm = Konko(model="meta-llama/Llama-2-13b-chat-hf")
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response = llm.complete(prompt)
        assert response.text is not None

        chat_response = llm.chat([message])
        assert chat_response.message.content is not None


def test_chat_model_basic_openai_model() -> None:
        llm = Konko(model="gpt-3.5-turbo")
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response = llm.complete(prompt)
        assert response.text is not None

        chat_response = llm.chat([message])
        assert chat_response.message.content is not None


def test_chat_model_streaming() -> None:
    llm = Konko(model="meta-llama/Llama-2-13b-chat-hf")
    message = ChatMessage(role="user", content="test message")
    chat_response_gen = llm.stream_chat([message])
    chat_responses = list(chat_response_gen)
    assert chat_responses[-1].message.content is not None