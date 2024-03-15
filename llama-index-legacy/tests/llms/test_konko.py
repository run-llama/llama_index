import pytest
from llama_index.legacy.core.llms.types import ChatMessage
from llama_index.legacy.llms.konko import Konko

try:
    import konko
except ImportError:
    konko = None  # type: ignore


@pytest.mark.skipif(konko is None, reason="konko not installed")
def test_chat_model_basic_non_openai_model() -> None:
    llm = Konko(model="meta-llama/llama-2-13b-chat")
    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    response = llm.complete(prompt)
    assert response.text is not None

    chat_response = llm.chat([message])
    assert chat_response.message.content is not None


@pytest.mark.skipif(konko is None, reason="konko not installed")
def test_chat_model_basic_openai_model() -> None:
    llm = Konko(model="gpt-3.5-turbo")
    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    response = llm.complete(prompt)
    assert response.text is not None

    chat_response = llm.chat([message])
    assert chat_response.message.content is not None


@pytest.mark.skipif(konko is None, reason="konko not installed")
def test_chat_model_streaming() -> None:
    llm = Konko(model="meta-llama/llama-2-13b-chat")
    message = ChatMessage(role="user", content="test message")
    chat_response_gen = llm.stream_chat([message])
    chat_responses = list(chat_response_gen)
    assert chat_responses[-1].message.content is not None


def teardown_module() -> None:
    import os

    del os.environ["KONKO_API_KEY"]
