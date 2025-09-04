import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from llama_index.llms.modelscope.base import ModelScopeLLM


@pytest.fixture()
def modelscope_llm():
    return ModelScopeLLM()


@pytest.fixture()
def prompt():
    return "Hi, my name is"


@pytest.fixture()
def messages():
    return [
        ChatMessage(content="Which movie is the best?"),
        ChatMessage(content="It's Die Hard for sure.", role=MessageRole.ASSISTANT),
        ChatMessage(content="Can you explain why?"),
    ]


@pytest.mark.complete
def test_modelscope_complete(modelscope_llm, prompt):
    response = modelscope_llm.complete(prompt)
    assert response is not None
    assert str(response).strip() != ""
    print(response)


@pytest.mark.complete
def test_modelscope_stream_complete(modelscope_llm, prompt):
    response = modelscope_llm.stream_complete(prompt)
    assert response is not None
    for r in response:
        assert r is not None
        assert str(r).strip() != ""
        print(r)


@pytest.mark.xfail(reason="20 is the default max_length of the generation config")
def test_modelscope_chat_clear(modelscope_llm, messages):
    response = modelscope_llm.chat(messages)
    assert response is not None
    assert str(response).strip() != ""
    print(response)


@pytest.mark.chat
def test_modelscope_chat(modelscope_llm, messages):
    response = modelscope_llm.chat(messages, max_new_tokens=100)
    assert response is not None
    assert str(response).strip() != ""
    print(response)


@pytest.mark.chat
def test_modelscope_stream_chat(modelscope_llm, messages):
    gen = modelscope_llm.stream_chat(messages, max_new_tokens=100)
    assert gen is not None
    for r in gen:
        assert r is not None
        assert str(r).strip() != ""
        print(r)
