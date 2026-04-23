from typing import List
from unittest.mock import MagicMock, patch

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai import Tokenizer
from llama_index.llms.openai_like import OpenAILikeResponses
from openai.types.responses import Response, ResponseOutputMessage, ResponseOutputText
from openai.types.responses.response import ResponseTextConfig
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)


class StubTokenizer(Tokenizer):
    def encode(self, text: str) -> List[int]:
        return [sum(ord(letter) for letter in word) for word in text.split(" ")]


STUB_MODEL_NAME = "models/stub-responses"
STUB_API_KEY = "stub_key"


def mock_response(text: str) -> Response:
    return Response(
        id="resp-abc123",
        object="response",
        created_at=1677858242,
        model=STUB_MODEL_NAME,
        output=[
            ResponseOutputMessage(
                id="msg-abc123",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(
                        type="output_text",
                        text=text,
                        annotations=[],
                    )
                ],
            )
        ],
        usage=ResponseUsage(
            input_tokens=13,
            output_tokens=7,
            total_tokens=20,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        ),
        tool_choice="auto",
        top_p=1.0,
        truncation="disabled",
        status="completed",
        text=ResponseTextConfig(),
        parallel_tool_calls=True,
        temperature=0.1,
        max_output_tokens=None,
        instructions=None,
        tools=[],
    )


def test_interfaces() -> None:
    llm = OpenAILikeResponses(model=STUB_MODEL_NAME, api_key=STUB_API_KEY)
    assert llm.class_name() == "OpenAILikeResponses"
    assert llm.model == STUB_MODEL_NAME


def test_metadata_defaults() -> None:
    llm = OpenAILikeResponses(model=STUB_MODEL_NAME, api_key=STUB_API_KEY)
    metadata = llm.metadata
    assert metadata.is_chat_model is True
    assert metadata.is_function_calling_model is False
    assert metadata.model_name == STUB_MODEL_NAME


def test_metadata_custom() -> None:
    llm = OpenAILikeResponses(
        model=STUB_MODEL_NAME,
        api_key=STUB_API_KEY,
        context_window=128000,
        is_function_calling_model=True,
    )
    metadata = llm.metadata
    assert metadata.context_window == 128000
    assert metadata.is_function_calling_model is True


def test_tokenizer_none() -> None:
    llm = OpenAILikeResponses(model=STUB_MODEL_NAME, api_key=STUB_API_KEY)
    assert llm._tokenizer is None


def test_tokenizer_instance() -> None:
    tok = StubTokenizer()
    llm = OpenAILikeResponses(
        model=STUB_MODEL_NAME, api_key=STUB_API_KEY, tokenizer=tok
    )
    assert llm._tokenizer is tok


@patch("llama_index.llms.openai.responses.SyncOpenAI")
def test_chat(MockSyncOpenAI: MagicMock) -> None:
    content = "hello from responses"
    mock_instance = MockSyncOpenAI.return_value
    mock_instance.responses.create.return_value = mock_response(content)

    llm = OpenAILikeResponses(
        model=STUB_MODEL_NAME,
        api_key=STUB_API_KEY,
        api_base="http://localhost:8080/v1",
        is_function_calling_model=True,
    )

    response = llm.chat([ChatMessage(role=MessageRole.USER, content="test message")])
    assert response.message.content == content
    mock_instance.responses.create.assert_called_once()


def test_serialization() -> None:
    llm = OpenAILikeResponses(
        model=STUB_MODEL_NAME,
        api_key=STUB_API_KEY,
        context_window=128000,
        is_function_calling_model=True,
        max_output_tokens=4096,
    )

    serialized = llm.to_dict()
    assert serialized["context_window"] == 128000
    assert serialized["is_function_calling_model"] is True
    assert serialized["max_output_tokens"] == 4096
