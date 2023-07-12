from llama_index.llms.anthropic import Anthropic
from llama_index.llms.base import ChatMessage


def test_basic() -> None:
    llm = Anthropic(model="test")
    test_prompt = "test prompt"
    response = llm.complete(test_prompt)
    assert len(response.text) > 0

    message = ChatMessage(role="user", content=test_prompt)
    chat_response = llm.chat([message])
    assert len(chat_response.message.content) > 0


def test_streaming() -> None:
    llm = Anthropic(model="test")
    test_prompt = "test prompt"
    response_gen = llm.stream_complete(test_prompt)
    responses = list(response_gen)
    assert len(responses[-1].text) > 0

    message = ChatMessage(role="user", content=test_prompt)
    chat_response_gen = llm.stream_chat([message])
    chat_responses = list(chat_response_gen)
    assert len(chat_responses[-1].message.content) > 0
