from typing import TypeVar, Iterable, AsyncIterator

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from gigachat.models import ChatCompletionChunk, ChoicesChunk, MessagesChunk
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage

from llama_index.llms.gigachat import GigaChatLLM, GigaChatModel


def test_text_inference_embedding_class():
    names_of_base_classes = [b.__name__ for b in GigaChatLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_gigachatllm_initialization():
    llm = GigaChatLLM()
    assert llm.model == GigaChatModel.GIGACHAT
    assert llm.context_window == 8192

    llm_plus = GigaChatLLM(model=GigaChatModel.GIGACHAT_PLUS)
    assert llm_plus.model == GigaChatModel.GIGACHAT_PLUS
    assert llm_plus.context_window == 32768


@patch("llama_index.llms.gigachat.base.GigaChat")
def test_complete(mock_gigachat):
    # Arrange
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Paris"
    mock_gigachat.return_value.__enter__.return_value.chat.return_value = mock_response

    llm = GigaChatLLM()

    # Act
    response = llm.complete("What is the capital of France?")

    # Assert
    assert response.text == "Paris"
    mock_gigachat.return_value.__enter__.return_value.chat.assert_called_once()


@patch("llama_index.llms.gigachat.base.GigaChat")
def test_chat(mock_gigachat):
    # Arrange
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello!"
    mock_gigachat.return_value.__enter__.return_value.chat.return_value = mock_response

    llm = GigaChatLLM()

    # Act
    response = llm.chat([ChatMessage(role="user", content="Hi")])

    # Assert
    assert response.message.content == "Hello!"
    mock_gigachat.return_value.__enter__.return_value.chat.assert_called_once()


@patch("llama_index.llms.gigachat.base.GigaChat")
@pytest.mark.asyncio
async def test_acomplete(mock_gigachat):
    # Arrange
    mock_response = AsyncMock()
    mock_response.choices[0].message.content = "Paris"
    mock_gigachat.return_value.__aenter__.return_value.achat.return_value = (
        mock_response
    )

    llm = GigaChatLLM()

    # Act
    response = await llm.acomplete("What is the capital of France?")

    # Assert
    assert response.text == "Paris"
    mock_gigachat.return_value.__aenter__.return_value.achat.assert_called_once()


@patch("llama_index.llms.gigachat.base.GigaChat")
@pytest.mark.asyncio
async def test_achat(mock_gigachat):
    # Arrange
    mock_response = AsyncMock()
    mock_response.choices[0].message.content = "Hello!"
    mock_gigachat.return_value.__aenter__.return_value.achat.return_value = (
        mock_response
    )

    llm = GigaChatLLM()

    # Act
    response = await llm.achat([ChatMessage(role="user", content="Hi")])

    # Assert
    assert response.message.content == "Hello!"
    mock_gigachat.return_value.__aenter__.return_value.achat.assert_called_once()


@patch("llama_index.llms.gigachat.base.GigaChat")
def test_stream_complete(mock_gigachat):
    # Arrange
    mock_gigachat_instance = mock_gigachat.return_value.__enter__.return_value
    mock_gigachat_instance.stream.return_value = iter(
        [
            ChatCompletionChunk(
                choices=[
                    ChoicesChunk(
                        delta=MessagesChunk(
                            content="Pa",
                        ),
                        index=0,
                    )
                ],
                created=1,
                model="gigachat",
                object="stream",
            ),
            ChatCompletionChunk(
                choices=[
                    ChoicesChunk(
                        delta=MessagesChunk(
                            content="ris",
                        ),
                        index=1,
                    )
                ],
                created=2,
                model="gigachat",
                object="stream",
            ),
        ]
    )

    llm = GigaChatLLM()

    # Act
    response = "".join(
        [resp.delta for resp in llm.stream_complete("What is the capital of France?")]
    )

    # Assert
    assert response == "Paris"
    mock_gigachat_instance.stream.assert_called_once()


T = TypeVar("T")


@patch("llama_index.llms.gigachat.base.GigaChat")
@pytest.mark.asyncio
async def test_astream_complete(mock_gigachat):
    # Arrange

    class AsyncIterWrapper(AsyncIterator[T]):
        def __init__(self, iterable: Iterable[T]) -> None:
            self.iterable = iter(iterable)

        def __aiter__(self) -> "AsyncIterWrapper":
            return self

        async def __anext__(self) -> T:
            try:
                return next(self.iterable)
            except StopIteration:
                raise StopAsyncIteration

    # Arrange
    mock_gigachat_instance = mock_gigachat.return_value.__aenter__.return_value

    # Mock a coroutine that returns an async iterable
    def mock_astream(chat):
        return AsyncIterWrapper(
            iter(
                [
                    ChatCompletionChunk(
                        choices=[
                            ChoicesChunk(
                                delta=MessagesChunk(
                                    content="Pa",
                                ),
                                index=0,
                            )
                        ],
                        created=1,
                        model="gigachat",
                        object="stream",
                    ),
                    ChatCompletionChunk(
                        choices=[
                            ChoicesChunk(
                                delta=MessagesChunk(
                                    content="ris",
                                ),
                                index=1,
                            )
                        ],
                        created=2,
                        model="gigachat",
                        object="stream",
                    ),
                ]
            )
        )

    mock_gigachat_instance.astream = mock_astream

    llm = GigaChatLLM()

    # Act
    response = ""
    async for resp in await llm.astream_complete("What is the capital of France?"):
        response += resp.delta

    # Assert
    assert response == "Paris"
