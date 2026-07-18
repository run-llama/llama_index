from typing import Any, AsyncGenerator, Generator, List
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.llms import ChatMessage
from llama_index.llms.deepseek import DeepSeek
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in DeepSeek.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def mock_chat_completion(text: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-deepseek123",
        object="chat.completion",
        created=1677858242,
        model="deepseek-chat",
        usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        choices=[
            Choice(
                message=ChatCompletionMessage(role="assistant", content=text),
                finish_reason="stop",
                index=0,
            )
        ],
    )


def mock_chat_completion_chunks(text_list: List[str]) -> List[ChatCompletionChunk]:
    return [
        ChatCompletionChunk(
            id="chatcmpl-deepseek123",
            object="chat.completion.chunk",
            created=1677858242,
            model="deepseek-chat",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content=text, role="assistant"),
                    finish_reason=None if i < len(text_list) - 1 else "stop",
                    index=0,
                )
            ],
        )
        for i, text in enumerate(text_list)
    ]


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_deepseek_chat(MockSyncOpenAI: MagicMock) -> None:
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion("DeepSeek is awesome!")
    
    llm = DeepSeek(model="deepseek-chat", api_key="fake_key")
    messages = [ChatMessage(role="user", content="Hello")]
    
    response = llm.chat(messages)
    assert response.message.content == "DeepSeek is awesome!"
    assert mock_instance.chat.completions.create.called


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_deepseek_complete(MockSyncOpenAI: MagicMock) -> None:
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = mock_chat_completion("DeepSeek is indeed awesome!")
    
    llm = DeepSeek(model="deepseek-chat", api_key="fake_key")
    response = llm.complete("Is DeepSeek awesome?")
    
    assert response.text == "DeepSeek is indeed awesome!"
    assert mock_instance.chat.completions.create.called


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_deepseek_stream_chat(MockSyncOpenAI: MagicMock) -> None:
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = iter(
        mock_chat_completion_chunks(["Deep", "Seek", " is", " fast"])
    )
    
    llm = DeepSeek(model="deepseek-chat", api_key="fake_key")
    messages = [ChatMessage(role="user", content="Stream chat test")]
    
    response_gen = llm.stream_chat(messages)
    responses = list(response_gen)
    
    assert responses[-1].message.content == "DeepSeek is fast"
    assert mock_instance.chat.completions.create.called


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_deepseek_stream_complete(MockSyncOpenAI: MagicMock) -> None:
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.chat.completions.create.return_value = iter(
        mock_chat_completion_chunks(["Deep", "Seek", " streams", " text"])
    )
    
    llm = DeepSeek(model="deepseek-chat", api_key="fake_key")
    response_gen = llm.stream_complete("Stream complete test")
    responses = list(response_gen)
    
    assert responses[-1].text == "DeepSeek streams text"
    assert mock_instance.chat.completions.create.called


@pytest.mark.asyncio()
@patch("llama_index.llms.openai.base.AsyncOpenAI")
async def test_deepseek_achat(MockAsyncOpenAI: MagicMock) -> None:
    mock_instance = MockAsyncOpenAI.return_value
    mock_instance.chat.completions.create = AsyncMock(return_value=mock_chat_completion("DeepSeek async response"))
    
    llm = DeepSeek(model="deepseek-chat", api_key="fake_key")
    messages = [ChatMessage(role="user", content="Async chat test")]
    
    response = await llm.achat(messages)
    assert response.message.content == "DeepSeek async response"


@pytest.mark.asyncio()
@patch("llama_index.llms.openai.base.AsyncOpenAI")
async def test_deepseek_acomplete(MockAsyncOpenAI: MagicMock) -> None:
    mock_instance = MockAsyncOpenAI.return_value
    mock_instance.chat.completions.create = AsyncMock(return_value=mock_chat_completion("DeepSeek async completion"))
    
    llm = DeepSeek(model="deepseek-chat", api_key="fake_key")
    response = await llm.acomplete("Async complete test")
    
    assert response.text == "DeepSeek async completion"


@pytest.mark.asyncio()
@patch("llama_index.llms.openai.base.AsyncOpenAI")
async def test_deepseek_astream_chat(MockAsyncOpenAI: MagicMock) -> None:
    mock_instance = MockAsyncOpenAI.return_value
    
    async def mock_async_generator(*args, **kwargs) -> AsyncGenerator[ChatCompletionChunk, None]:
        for chunk in mock_chat_completion_chunks(["Deep", "Seek", " async", " stream"]):
            yield chunk

    mock_instance.chat.completions.create = AsyncMock(side_effect=mock_async_generator)
    
    llm = DeepSeek(model="deepseek-chat", api_key="fake_key")
    messages = [ChatMessage(role="user", content="Async stream chat test")]
    
    response_gen = await llm.astream_chat(messages)
    responses = [item async for item in response_gen]
    
    assert responses[-1].message.content == "DeepSeek async stream"


@pytest.mark.asyncio()
@patch("llama_index.llms.openai.base.AsyncOpenAI")
async def test_deepseek_astream_complete(MockAsyncOpenAI: MagicMock) -> None:
    mock_instance = MockAsyncOpenAI.return_value
    
    async def mock_async_generator(*args, **kwargs) -> AsyncGenerator[ChatCompletionChunk, None]:
        for chunk in mock_chat_completion_chunks(["Deep", "Seek", " async", " complete", " stream"]):
            yield chunk

    mock_instance.chat.completions.create = AsyncMock(side_effect=mock_async_generator)
    
    llm = DeepSeek(model="deepseek-chat", api_key="fake_key")
    response_gen = await llm.astream_complete("Async stream complete test")
    responses = [item async for item in response_gen]
    
    assert responses[-1].text == "DeepSeek async complete stream"
