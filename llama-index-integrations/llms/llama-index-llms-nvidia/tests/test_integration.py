import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
from llama_index.llms.nvidia import NVIDIA


@pytest.mark.integration
def test_chat(chat_model: str, mode: dict) -> None:
    message = ChatMessage(content="Hello")
    response = NVIDIA(model=chat_model, **mode).chat([message])
    assert isinstance(response, ChatResponse)
    assert isinstance(response.message, ChatMessage)
    assert isinstance(response.message.content, str)


@pytest.mark.integration
def test_complete(chat_model: str, mode: dict) -> None:
    response = NVIDIA(model=chat_model, **mode).complete("Hello")
    assert isinstance(response, CompletionResponse)
    assert isinstance(response.text, str)


@pytest.mark.integration
def test_stream_chat(chat_model: str, mode: dict) -> None:
    message = ChatMessage(content="Hello")
    gen = NVIDIA(model=chat_model, **mode).stream_chat([message])
    assert all(isinstance(response, ChatResponse) for response in gen)
    assert all(isinstance(response.delta, str) for response in gen)


@pytest.mark.integration
def test_stream_complete(chat_model: str, mode: dict) -> None:
    gen = NVIDIA(model=chat_model, **mode).stream_complete("Hello")
    assert all(isinstance(response, CompletionResponse) for response in gen)
    assert all(isinstance(response.delta, str) for response in gen)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_achat(chat_model: str, mode: dict) -> None:
    message = ChatMessage(content="Hello")
    response = await NVIDIA(model=chat_model, **mode).achat([message])
    assert isinstance(response, ChatResponse)
    assert isinstance(response.message, ChatMessage)
    assert isinstance(response.message.content, str)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acomplete(chat_model: str, mode: dict) -> None:
    response = await NVIDIA(model=chat_model, **mode).acomplete("Hello")
    assert isinstance(response, CompletionResponse)
    assert isinstance(response.text, str)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_astream_chat(chat_model: str, mode: dict) -> None:
    message = ChatMessage(content="Hello")
    gen = await NVIDIA(model=chat_model, **mode).astream_chat([message])
    responses = [response async for response in gen]
    assert all(isinstance(response, ChatResponse) for response in responses)
    assert all(isinstance(response.delta, str) for response in responses)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_astream_complete(chat_model: str, mode: dict) -> None:
    gen = await NVIDIA(model=chat_model, **mode).astream_complete("Hello")
    responses = [response async for response in gen]
    assert all(isinstance(response, CompletionResponse) for response in responses)
    assert all(isinstance(response.delta, str) for response in responses)


@pytest.mark.integration
@pytest.mark.parametrize(
    "excluded",
    [
        "mistralai/mixtral-8x22b-v0.1",  # not a /chat/completion endpoint
    ],
)
def test_exclude_models(mode: dict, excluded: str) -> None:
    assert excluded not in [model.id for model in NVIDIA(**mode).available_models]
