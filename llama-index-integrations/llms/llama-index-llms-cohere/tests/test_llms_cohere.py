from typing import Sequence, Optional, List
from unittest import mock

import pytest
from cohere import NonStreamedChatResponse

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatResponse, ChatMessage, MessageRole
from llama_index.core.llms.mock import MockLLM

from llama_index.llms.cohere import Cohere, DocumentMessage, is_cohere_model


def test_is_cohere():
    assert is_cohere_model(Cohere(api_key="mario"))
    assert not is_cohere_model(MockLLM())


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in Cohere.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@pytest.mark.parametrize(
    "messages,expected_chat_history,expected_documents,expected_message",  # noqa: PT006
    [
        pytest.param(
            [ChatMessage(content="Hello", role=MessageRole.USER)],
            [],
            None,
            "Hello",
            id="single user message",
        ),
        pytest.param(
            [
                ChatMessage(content="Earliest message", role=MessageRole.USER),
                ChatMessage(content="Latest message", role=MessageRole.USER),
            ],
            [{"message": "Earliest message", "role": "USER"}],
            None,
            "Latest message",
            id="messages with chat history",
        ),
        pytest.param(
            [
                ChatMessage(content="Earliest message", role=MessageRole.USER),
                DocumentMessage(content="Document content"),
                ChatMessage(content="Latest message", role=MessageRole.USER),
            ],
            [{"message": "Earliest message", "role": "USER"}],
            [{"text": "Document content"}],
            "Latest message",
            id="messages with chat history",
        ),
    ],
)
def test_chat(
    messages: Sequence[ChatMessage],
    expected_chat_history: Optional[List],
    expected_documents: Optional[List],
    expected_message: str,
):
    # Mock the API client.
    with mock.patch("llama_index.llms.cohere.base.cohere.Client", autospec=True):
        llm = Cohere(api_key="dummy", temperature=0.3)
    # Mock the API response.
    llm._client.chat.return_value = NonStreamedChatResponse(text="Placeholder reply")
    expected = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="Placeholder reply"),
        raw=llm._client.chat.return_value.__dict__,
    )

    actual = llm.chat(messages)

    assert expected == actual
    # Assert that the mocked API client was called in the expected way.
    llm._client.chat.assert_called_once_with(
        chat_history=expected_chat_history,
        documents=expected_documents,
        message=expected_message,
        model="command-r",
        temperature=0.3,
    )
