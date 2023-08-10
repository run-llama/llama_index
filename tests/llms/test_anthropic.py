import pytest
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.base import ChatMessage

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore


@pytest.mark.skipif(anthropic is None, reason="anthropic not installed")
def test_basic() -> None:
    llm = Anthropic(model="claude-instant-1")
    test_prompt = "test prompt"
    response = llm.complete(test_prompt)
    assert len(response.text) > 0

    message = ChatMessage(role="user", content=test_prompt)
    chat_response = llm.chat([message])
    assert chat_response.message.content is not None
    assert len(chat_response.message.content) > 0


@pytest.mark.skipif(anthropic is None, reason="anthropic not installed")
def test_streaming() -> None:
    llm = Anthropic(model="claude-instant-1")
    test_prompt = "test prompt"
    response_gen = llm.stream_complete(test_prompt)
    for r in response_gen:
        assert r.delta is not None
        assert r.text is not None

    message = ChatMessage(role="user", content=test_prompt)
    chat_response_gen = llm.stream_chat([message])
    for r_ in chat_response_gen:
        assert r_.message.content is not None
        assert r_.delta is not None


@pytest.mark.skipif(anthropic is None, reason="anthropic not installed")
@pytest.mark.asyncio
async def test_async() -> None:
    llm = Anthropic(model="claude-instant-1")
    test_prompt = "test prompt"
    response = await llm.acomplete(test_prompt)
    assert len(response.text) > 0

    message = ChatMessage(role="user", content=test_prompt)
    chat_response = await llm.achat([message])
    assert chat_response.message.content is not None
    assert len(chat_response.message.content) > 0


@pytest.mark.skipif(anthropic is None, reason="anthropic not installed")
@pytest.mark.asyncio
async def test_async_streaming() -> None:
    llm = Anthropic(model="test")
    test_prompt = "test prompt"
    response_gen = await llm.astream_complete(test_prompt)
    async for r in response_gen:
        assert r.delta is not None
        assert r.text is not None

    message = ChatMessage(role="user", content=test_prompt)
    chat_response_gen = await llm.astream_chat([message])
    async for r_ in chat_response_gen:
        assert r_.message.content is not None
        assert r_.delta is not None
