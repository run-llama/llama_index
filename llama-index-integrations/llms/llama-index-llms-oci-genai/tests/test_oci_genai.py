# Test OCI Generative AI LLM service

from unittest.mock import MagicMock
from typing import Any

import pytest
from pytest import MonkeyPatch

from llama_index.llms.oci_genai import OCIGenAI
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    TextBlock,
    ImageBlock,
)
from llama_index.core.tools import FunctionTool
import json


class MockResponseDict(dict):
    def __getattr__(self, val) -> Any:  # type: ignore[no-untyped-def]
        return self[val]


@pytest.mark.parametrize("test_model_id", [])
def test_llm_complete(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test valid completion call to OCI Generative AI LLM service."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    provider = llm._provider.__class__.__name__

    def mocked_response(*args):  # type: ignore[no-untyped-def]
        response_text = "This is the completion."

        if provider == "CohereProvider":
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "inference_response": MockResponseDict(
                                {
                                    "generated_texts": [
                                        MockResponseDict(
                                            {
                                                "text": response_text,
                                            }
                                        )
                                    ]
                                }
                            )
                        }
                    ),
                }
            )
        elif provider == "MetaProvider" or provider == "XAIProvider":
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "inference_response": MockResponseDict(
                                {
                                    "choices": [
                                        MockResponseDict(
                                            {
                                                "text": response_text,
                                            }
                                        )
                                    ]
                                }
                            )
                        }
                    ),
                }
            )
        else:
            return None

    monkeypatch.setattr(llm._client, "generate_text", mocked_response)

    output = llm.complete("This is a prompt.", temperature=0.2)
    assert output.text == "This is the completion."


@pytest.mark.parametrize(
    "test_model_id",
    [
        "cohere.command-r-16k",
        "cohere.command-r-plus",
        "meta.llama-3-70b-instruct",
        "meta.llama-3.1-70b-instruct",
        "xai.grok-3-mini",
        "xai.grok-4-fast-non-reasoning",
    ],
)
def test_llm_chat(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test valid chat call to OCI Generative AI LLM service."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    provider = llm._provider.__class__.__name__

    def mocked_response(*args):  # type: ignore[no-untyped-def]
        response_text = "Assistant chat reply."
        response = None
        if provider == "CohereProvider":
            response = MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "text": response_text,
                                    "finish_reason": "stop",
                                    "documents": [],
                                    "citations": [],
                                    "search_queries": [],
                                    "is_search_required": False,
                                    "tool_calls": None,
                                }
                            ),
                            "model_id": "cohere.command-r-16k",
                            "model_version": "1.0",
                        }
                    ),
                    "request_id": "req-1234567890",
                    "headers": {"content-length": "1234"},
                }
            )
        elif provider == "MetaProvider" or provider == "XAIProvider":
            response = MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "choices": [
                                        MockResponseDict(
                                            {
                                                "message": MockResponseDict(
                                                    {
                                                        "content": [
                                                            MockResponseDict(
                                                                {
                                                                    "text": response_text,
                                                                }
                                                            )
                                                        ]
                                                    }
                                                ),
                                                "finish_reason": "stop",
                                            }
                                        )
                                    ],
                                    "time_created": "2024-11-03T12:00:00Z",
                                }
                            ),
                            "model_id": "meta.llama-3-70b-instruct",
                            "model_version": "1.0",
                        }
                    ),
                    "request_id": "req-0987654321",
                    "headers": {"content-length": "1234"},
                }
            )
        return response

    monkeypatch.setattr(llm._client, "chat", mocked_response)

    messages = [
        ChatMessage(role="user", content="User message"),
    ]

    # For Meta provider, we expect fewer fields in additional_kwargs
    if provider == "MetaProvider" or provider == "XAIProvider":
        additional_kwargs = {
            "finish_reason": "stop",
            "time_created": "2024-11-03T12:00:00Z",
        }
    else:
        additional_kwargs = {
            "finish_reason": "stop",
            "documents": [],
            "citations": [],
            "search_queries": [],
            "is_search_required": False,
        }

    expected = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Assistant chat reply.",
            additional_kwargs=additional_kwargs,
        ),
        raw={},  # Mocked raw data
        additional_kwargs={
            "model_id": test_model_id,
            "model_version": "1.0",
            "request_id": "req-1234567890"
            if test_model_id == "cohere.command-r-16k"
            else "req-0987654321",
            "content-length": "1234",
        },
    )

    actual = llm.chat(messages, temperature=0.2)
    assert actual.message.content == expected.message.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_model_id",
    ["cohere.command-r-16k", "meta.llama-3-70b-instruct"],
)
async def test_llm_achat(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test async chat (achat) call to OCI Generative AI LLM service."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    provider = llm._provider.__class__.__name__

    def mocked_response(*args):  # type: ignore[no-untyped-def]
        response_text = "Async assistant reply."
        if provider == "CohereProvider":
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "text": response_text,
                                    "finish_reason": "stop",
                                    "documents": [],
                                    "citations": [],
                                    "search_queries": [],
                                    "is_search_required": False,
                                    "tool_calls": None,
                                }
                            ),
                            "model_id": test_model_id,
                            "model_version": "1.0",
                        }
                    ),
                    "request_id": "req-async-123",
                    "headers": {"content-length": "1234"},
                }
            )
        else:
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "choices": [
                                        MockResponseDict(
                                            {
                                                "message": MockResponseDict(
                                                    {
                                                        "content": [
                                                            MockResponseDict(
                                                                {"text": response_text}
                                                            )
                                                        ]
                                                    }
                                                ),
                                                "finish_reason": "stop",
                                            }
                                        )
                                    ],
                                    "time_created": "2024-11-03T12:00:00Z",
                                }
                            ),
                            "model_id": test_model_id,
                            "model_version": "1.0",
                        }
                    ),
                    "request_id": "req-async-456",
                    "headers": {"content-length": "1234"},
                }
            )

    monkeypatch.setattr(llm._client, "chat", mocked_response)

    messages = [ChatMessage(role="user", content="User message")]
    actual = await llm.achat(messages, temperature=0.2)
    assert actual.message.content == "Async assistant reply."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_model_id",
    ["cohere.command-r-16k", "meta.llama-3-70b-instruct"],
)
async def test_llm_acomplete(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test async complete (acomplete) call to OCI Generative AI LLM service."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    provider = llm._provider.__class__.__name__

    def mocked_response(*args):  # type: ignore[no-untyped-def]
        response_text = "Async completion text."
        if provider == "CohereProvider":
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "text": response_text,
                                    "finish_reason": "stop",
                                    "documents": [],
                                    "citations": [],
                                    "search_queries": [],
                                    "is_search_required": False,
                                    "tool_calls": None,
                                }
                            ),
                            "model_id": test_model_id,
                            "model_version": "1.0",
                        }
                    ),
                    "request_id": "req-acomplete-123",
                    "headers": {"content-length": "1234"},
                }
            )
        else:
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "choices": [
                                        MockResponseDict(
                                            {
                                                "message": MockResponseDict(
                                                    {
                                                        "content": [
                                                            MockResponseDict(
                                                                {"text": response_text}
                                                            )
                                                        ]
                                                    }
                                                ),
                                                "finish_reason": "stop",
                                            }
                                        )
                                    ],
                                    "time_created": "2024-11-03T12:00:00Z",
                                }
                            ),
                            "model_id": test_model_id,
                            "model_version": "1.0",
                        }
                    ),
                    "request_id": "req-acomplete-456",
                    "headers": {"content-length": "1234"},
                }
            )

    monkeypatch.setattr(llm._client, "chat", mocked_response)

    actual = await llm.acomplete("Hello", temperature=0.2)
    assert actual.text == "Async completion text."


@pytest.mark.parametrize(
    "test_model_id", ["cohere.command-r-16k", "cohere.command-r-plus", "xai.grok-4"]
)
def test_llm_chat_with_tools(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test chat_with_tools call to OCI Generative AI LLM service with tool calling."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    provider = llm._provider.__class__.__name__

    def mock_tool_function(param1: str) -> str:
        """Mock tool function that takes a string parameter."""
        return f"Mock tool function called with {param1}"

    # Create proper FunctionTool
    mock_tool = FunctionTool.from_defaults(fn=mock_tool_function)
    tools = [mock_tool]

    messages = [
        ChatMessage(role="user", content="User message"),
    ]

    # Mock the client response
    def mocked_response(*args, **kwargs):
        response_text = "Assistant chat reply."
        tool_calls = []
        if provider == "CohereProvider":
            tool_calls = [
                MockResponseDict(
                    {
                        "name": "mock_tool_function",
                        "parameters": {"param1": "test"},
                    }
                )
            ]
        elif provider == "XAIProvider":
            tool_calls = [
                MockResponseDict(
                    {
                        "arguments": '{"param1": "test"}',
                        "name": "mock_tool_function",
                        "id": "call_38131587",
                    }
                )
            ]

        response = None
        if provider == "CohereProvider":
            response = MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "text": response_text,
                                    "finish_reason": "stop",
                                    "documents": [],
                                    "citations": [],
                                    "search_queries": [],
                                    "is_search_required": False,
                                    "tool_calls": tool_calls,
                                }
                            ),
                            "model_id": test_model_id,
                            "model_version": "1.0",
                        }
                    ),
                    "request_id": "req-1234567890",
                    "headers": {"content-length": "1234"},
                }
            )
        elif provider == "XAIProvider":
            response = MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "choices": [
                                        MockResponseDict(
                                            {
                                                "message": MockResponseDict(
                                                    {
                                                        "content": [
                                                            MockResponseDict(
                                                                {
                                                                    "text": "",
                                                                }
                                                            )
                                                        ],
                                                        "role": "ASSISTANT",
                                                        "tool_calls": tool_calls,
                                                    }
                                                ),
                                                "finish_reason": "tool_calls",
                                            }
                                        )
                                    ],
                                    "time_created": "2024-11-03T12:00:00Z",
                                }
                            ),
                            "model_id": test_model_id,
                            "model_version": "1.0",
                        }
                    ),
                    "request_id": "req-0987654321",
                    "headers": {"content-length": "1234"},
                }
            )

        else:
            # MetaProvider does not support tools
            raise NotImplementedError("Tools not supported for this provider.")
        return response

    monkeypatch.setattr(llm._client, "chat", mocked_response)

    actual_response = llm.chat(
        messages=messages,
        tools=tools,
    )

    # Expected response structure
    expected_tool_calls = []

    if provider == "CohereProvider":
        expected_tool_calls = [
            {
                "name": "mock_tool_function",
                "toolUseId": actual_response.message.additional_kwargs["tool_calls"][0][
                    "toolUseId"
                ],
                "input": json.dumps({"param1": "test"}),
            }
        ]
    elif provider == "XAIProvider":
        expected_tool_calls = [
            {
                "name": "mock_tool_function",
                "toolUseId": "1234",
                "input": json.dumps({"param1": "test"}),
            }
        ]

    expected_response = None
    if provider == "CohereProvider":
        expected_response = ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Assistant chat reply.",
                additional_kwargs={
                    "finish_reason": "stop",
                    "documents": [],
                    "citations": [],
                    "search_queries": [],
                    "is_search_required": False,
                    "tool_calls": expected_tool_calls,
                },
            ),
            raw={},
        )
    elif provider == "XAIProvider":
        expected_response = ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                additional_kwargs={
                    "finish_reason": "tool_calls",
                    "tool_calls": expected_tool_calls,
                },
            ),
            raw={},
        )

    # Compare everything except the toolUseId which is randomly generated
    assert actual_response.message.role == expected_response.message.role
    assert actual_response.message.content == expected_response.message.content

    actual_kwargs = actual_response.message.additional_kwargs
    expected_kwargs = expected_response.message.additional_kwargs

    # Check all non-tool_calls fields
    for key in [k for k in expected_kwargs if k != "tool_calls"]:
        assert actual_kwargs[key] == expected_kwargs[key]

    # Check tool calls separately
    actual_tool_calls = actual_kwargs["tool_calls"]
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tc, expected_tc in zip(actual_tool_calls, expected_tool_calls):
        assert actual_tc["name"] == expected_tc["name"]
        assert actual_tc["input"] == expected_tc["input"]
        assert "toolUseId" in actual_tc
        assert isinstance(actual_tc["toolUseId"], str)
        assert len(actual_tc["toolUseId"]) > 0

    # Check additional_kwargs
    assert actual_response.additional_kwargs == expected_response.additional_kwargs


@pytest.mark.parametrize(
    "test_model_id",
    ["meta.llama-3-70b-instruct", "meta.llama-3.1-70b-instruct", "xai.grok-4"],
)
def test_llm_multimodal_chat_with_image(
    monkeypatch: MonkeyPatch, test_model_id: str
) -> None:
    """Test multimodal chat call to OCI Generative AI LLM service with image input."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    def mocked_response(*args, **kwargs):
        response_text = "The image contains the OCI logo."
        return MockResponseDict(
            {
                "status": 200,
                "data": MockResponseDict(
                    {
                        "chat_response": MockResponseDict(
                            {
                                "choices": [
                                    MockResponseDict(
                                        {
                                            "message": MockResponseDict(
                                                {
                                                    "content": [
                                                        MockResponseDict(
                                                            {"text": response_text}
                                                        )
                                                    ]
                                                }
                                            ),
                                            "finish_reason": "stop",
                                        }
                                    )
                                ],
                                "time_created": "2024-07-02T12:00:00Z",
                            }
                        ),
                        "model_id": test_model_id,
                        "model_version": "1.0",
                    }
                ),
                "request_id": "req-0987654321",
                "headers": {"content-length": "1234"},
            }
        )

    monkeypatch.setattr(llm._client, "chat", mocked_response)

    image_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX///+/v7+jQ3Y5AAAADklEQVQI12P4AIX8EAgALgAD/aNpbtEAAAAASUVORK5CYII="
    messages = [
        ChatMessage(
            role="user",
            content=[
                TextBlock(text="What is in this image?"),
                ImageBlock(image_url=image_url),
            ],
        )
    ]

    expected = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content="The image contains the OCI logo.",
            additional_kwargs={
                "finish_reason": "stop",
                "time_created": "2024-07-02T12:00:00Z",
            },
        ),
        raw={},
        additional_kwargs={
            "model_id": test_model_id,
            "model_version": "1.0",
            "request_id": "req-0987654321",
            "content-length": "1234",
        },
    )

    actual = llm.chat(messages)

    assert actual.message.content == expected.message.content


# --- Stream event helpers for stream_chat / astream_chat tests ---


def _make_cohere_stream_events() -> list:
    """Create mock stream events for Cohere provider."""
    events_data = [
        {"text": "Stream "},
        {"text": "chunk "},
        {"text": "response", "finishReason": "stop"},
    ]
    result = []
    for data in events_data:
        event = type("Event", (), {})()
        event.data = json.dumps(data)
        result.append(event)
    return result


def _make_meta_stream_events() -> list:
    """Create mock stream events for Meta/XAI provider."""
    events_data = [
        {"message": {"content": [{"text": "Stream "}]}},
        {"message": {"content": [{"text": "chunk "}]}},
        {"message": {"content": [{"text": "response"}]}, "finishReason": "stop"},
    ]
    result = []
    for data in events_data:
        event = type("Event", (), {})()
        event.data = json.dumps(data)
        result.append(event)
    return result


@pytest.mark.parametrize(
    "test_model_id",
    ["cohere.command-r-16k", "meta.llama-3-70b-instruct"],
)
def test_llm_stream_chat(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test sync stream_chat call to OCI Generative AI LLM service."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    provider = llm._provider.__class__.__name__

    def mock_events():
        if provider == "CohereProvider":
            return _make_cohere_stream_events()
        else:
            return _make_meta_stream_events()

    def mocked_chat(*args, **kwargs):
        mock_data = MagicMock()
        mock_data.events = mock_events
        mock_response = MagicMock()
        mock_response.data = mock_data
        return mock_response

    monkeypatch.setattr(llm._client, "chat", mocked_chat)

    messages = [ChatMessage(role="user", content="User message")]
    chunks = list(llm.stream_chat(messages, temperature=0.2))

    assert len(chunks) >= 1
    # Final chunk has full accumulated content
    final_content = chunks[-1].message.content or ""
    assert (
        "Stream" in final_content
        and "chunk" in final_content
        and "response" in final_content
    )


@pytest.mark.parametrize(
    "test_model_id",
    ["cohere.command-r-16k", "meta.llama-3-70b-instruct"],
)
def test_llm_stream_complete(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test sync stream_complete call to OCI Generative AI LLM service."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    provider = llm._provider.__class__.__name__

    def mock_events():
        if provider == "CohereProvider":
            return _make_cohere_stream_events()
        else:
            return _make_meta_stream_events()

    def mocked_chat(*args, **kwargs):
        mock_data = MagicMock()
        mock_data.events = mock_events
        mock_response = MagicMock()
        mock_response.data = mock_data
        return mock_response

    monkeypatch.setattr(llm._client, "chat", mocked_chat)

    chunks = list(llm.stream_complete("Hello", temperature=0.2))

    assert len(chunks) >= 1
    final_text = chunks[-1].text or ""
    assert "Stream" in final_text and "chunk" in final_text and "response" in final_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_model_id",
    ["cohere.command-r-16k", "meta.llama-3-70b-instruct"],
)
async def test_llm_astream_chat(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test async astream_chat call to OCI Generative AI LLM service."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    provider = llm._provider.__class__.__name__

    def mock_events():
        if provider == "CohereProvider":
            return _make_cohere_stream_events()
        else:
            return _make_meta_stream_events()

    def mocked_chat(*args, **kwargs):
        mock_data = MagicMock()
        mock_data.events = mock_events
        mock_response = MagicMock()
        mock_response.data = mock_data
        return mock_response

    monkeypatch.setattr(llm._client, "chat", mocked_chat)

    messages = [ChatMessage(role="user", content="User message")]
    chunks = []
    astream = llm.astream_chat(messages, temperature=0.2)
    agen = await astream if hasattr(astream, "__await__") else astream
    async for chunk in agen:
        chunks.append(chunk)

    assert len(chunks) >= 1
    final_content = chunks[-1].message.content or ""
    assert (
        "Stream" in final_content
        and "chunk" in final_content
        and "response" in final_content
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_model_id",
    ["cohere.command-r-16k", "meta.llama-3-70b-instruct"],
)
async def test_llm_astream_complete(
    monkeypatch: MonkeyPatch, test_model_id: str
) -> None:
    """Test async astream_complete call to OCI Generative AI LLM service."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    provider = llm._provider.__class__.__name__

    def mock_events():
        if provider == "CohereProvider":
            return _make_cohere_stream_events()
        else:
            return _make_meta_stream_events()

    def mocked_chat(*args, **kwargs):
        mock_data = MagicMock()
        mock_data.events = mock_events
        mock_response = MagicMock()
        mock_response.data = mock_data
        return mock_response

    monkeypatch.setattr(llm._client, "chat", mocked_chat)

    chunks = []
    astream = llm.astream_complete("Hello", temperature=0.2)
    agen = await astream if hasattr(astream, "__await__") else astream
    async for chunk in agen:
        chunks.append(chunk)

    assert len(chunks) >= 1
    final_text = chunks[-1].text or ""
    assert "Stream" in final_text and "chunk" in final_text and "response" in final_text


@pytest.mark.parametrize(
    "test_model_id",
    [
        "cohere.command-r-16k",
        "cohere.command-r-plus",
        "meta.llama-3-70b-instruct",
        "xai.grok-3-mini",
    ],
)
def test_llm_complete_with_models(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test sync complete call with various model IDs (uses chat API)."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    provider = llm._provider.__class__.__name__

    def mocked_chat(*args, **kwargs):
        response_text = "Completion output for model."
        if provider == "CohereProvider":
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "text": response_text,
                                    "finish_reason": "stop",
                                    "documents": [],
                                    "citations": [],
                                    "search_queries": [],
                                    "is_search_required": False,
                                    "tool_calls": None,
                                }
                            ),
                            "model_id": test_model_id,
                            "model_version": "1.0",
                        }
                    ),
                    "request_id": "req-complete",
                    "headers": {"content-length": "1234"},
                }
            )
        else:
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "choices": [
                                        MockResponseDict(
                                            {
                                                "message": MockResponseDict(
                                                    {
                                                        "content": [
                                                            MockResponseDict(
                                                                {"text": response_text}
                                                            )
                                                        ]
                                                    }
                                                ),
                                                "finish_reason": "stop",
                                            }
                                        )
                                    ],
                                    "time_created": "2024-11-03T12:00:00Z",
                                }
                            ),
                            "model_id": test_model_id,
                            "model_version": "1.0",
                        }
                    ),
                    "request_id": "req-complete",
                    "headers": {"content-length": "1234"},
                }
            )

    monkeypatch.setattr(llm._client, "chat", mocked_chat)

    output = llm.complete("Test prompt", temperature=0.2)
    assert output.text == "Completion output for model."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_model_id",
    ["cohere.command-r-16k", "xai.grok-4"],
)
async def test_llm_achat_with_tools(
    monkeypatch: MonkeyPatch, test_model_id: str
) -> None:
    """Test async chat (achat) with tools to OCI Generative AI LLM service."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)
    provider = llm._provider.__class__.__name__

    def mock_tool_function(param1: str) -> str:
        """Mock tool function for testing."""
        return f"Result for {param1}"

    mock_tool = FunctionTool.from_defaults(fn=mock_tool_function)
    tools = [mock_tool]
    messages = [ChatMessage(role="user", content="User message")]

    def mocked_response(*args, **kwargs):
        response_text = "Async tool reply."
        tool_calls = []
        if provider == "CohereProvider":
            tool_calls = [
                MockResponseDict(
                    {"name": "mock_tool_function", "parameters": {"param1": "test"}}
                )
            ]
        elif provider == "XAIProvider":
            tool_calls = [
                MockResponseDict(
                    {
                        "arguments": '{"param1": "test"}',
                        "name": "mock_tool_function",
                        "id": "call_async_123",
                    }
                )
            ]

        if provider == "CohereProvider":
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "text": response_text,
                                    "finish_reason": "stop",
                                    "documents": [],
                                    "citations": [],
                                    "search_queries": [],
                                    "is_search_required": False,
                                    "tool_calls": tool_calls,
                                }
                            ),
                            "model_id": test_model_id,
                            "model_version": "1.0",
                        }
                    ),
                    "request_id": "req-async-tools",
                    "headers": {"content-length": "1234"},
                }
            )
        else:
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "choices": [
                                        MockResponseDict(
                                            {
                                                "message": MockResponseDict(
                                                    {
                                                        "content": [
                                                            MockResponseDict(
                                                                {"text": response_text}
                                                            )
                                                        ],
                                                        "role": "ASSISTANT",
                                                        "tool_calls": tool_calls,
                                                    }
                                                ),
                                                "finish_reason": "stop",
                                            }
                                        )
                                    ],
                                    "time_created": "2024-11-03T12:00:00Z",
                                }
                            ),
                            "model_id": test_model_id,
                            "model_version": "1.0",
                        }
                    ),
                    "request_id": "req-async-tools-xai",
                    "headers": {"content-length": "1234"},
                }
            )

    monkeypatch.setattr(llm._client, "chat", mocked_response)

    actual = await llm.achat(messages, tools=tools)
    assert actual.message.content == "Async tool reply."
    if provider == "CohereProvider":
        assert "tool_calls" in actual.message.additional_kwargs
        assert len(actual.message.additional_kwargs["tool_calls"]) == 1
        assert (
            actual.message.additional_kwargs["tool_calls"][0]["name"]
            == "mock_tool_function"
        )


@pytest.mark.parametrize(
    "test_model_id",
    ["cohere.command-r-16k", "meta.llama-3-70b-instruct", "xai.grok-4"],
)
def test_llm_chat_message_roles(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test chat with different message roles (system, user, assistant)."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    provider = llm._provider.__class__.__name__

    def mocked_response(*args):
        response_text = "Response to multi-role conversation."
        if provider == "CohereProvider":
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "text": response_text,
                                    "finish_reason": "stop",
                                    "documents": [],
                                    "citations": [],
                                    "search_queries": [],
                                    "is_search_required": False,
                                    "tool_calls": None,
                                }
                            ),
                            "model_id": test_model_id,
                            "model_version": "1.0",
                        }
                    ),
                    "request_id": "req-multi-role",
                    "headers": {"content-length": "1234"},
                }
            )
        else:
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "choices": [
                                        MockResponseDict(
                                            {
                                                "message": MockResponseDict(
                                                    {
                                                        "content": [
                                                            MockResponseDict(
                                                                {"text": response_text}
                                                            )
                                                        ]
                                                    }
                                                ),
                                                "finish_reason": "stop",
                                            }
                                        )
                                    ],
                                    "time_created": "2024-11-03T12:00:00Z",
                                }
                            ),
                            "model_id": test_model_id,
                            "model_version": "1.0",
                        }
                    ),
                    "request_id": "req-multi-role",
                    "headers": {"content-length": "1234"},
                }
            )

    monkeypatch.setattr(llm._client, "chat", mocked_response)

    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="First user message"),
        ChatMessage(role="assistant", content="Assistant reply"),
        ChatMessage(role="user", content="Second user message"),
    ]

    actual = llm.chat(messages, temperature=0.2)
    assert actual.message.content == "Response to multi-role conversation."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_model_id",
    ["cohere.command-r-16k", "meta.llama-3-70b-instruct"],
)
async def test_llm_acomplete_temperature_and_max_tokens(
    monkeypatch: MonkeyPatch, test_model_id: str
) -> None:
    """Test acomplete with custom temperature and max_tokens parameters."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    provider = llm._provider.__class__.__name__

    def mocked_response(*args):
        if provider == "CohereProvider":
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "text": "Custom params response.",
                                    "finish_reason": "stop",
                                    "documents": [],
                                    "citations": [],
                                    "search_queries": [],
                                    "is_search_required": False,
                                    "tool_calls": None,
                                }
                            ),
                            "model_id": test_model_id,
                            "model_version": "1.0",
                        }
                    ),
                    "request_id": "req-params",
                    "headers": {"content-length": "1234"},
                }
            )
        else:
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "choices": [
                                        MockResponseDict(
                                            {
                                                "message": MockResponseDict(
                                                    {
                                                        "content": [
                                                            MockResponseDict(
                                                                {
                                                                    "text": "Custom params response."
                                                                }
                                                            )
                                                        ]
                                                    }
                                                ),
                                                "finish_reason": "stop",
                                            }
                                        )
                                    ],
                                    "time_created": "2024-11-03T12:00:00Z",
                                }
                            ),
                            "model_id": test_model_id,
                            "model_version": "1.0",
                        }
                    ),
                    "request_id": "req-params",
                    "headers": {"content-length": "1234"},
                }
            )

    monkeypatch.setattr(llm._client, "chat", mocked_response)

    result = await llm.acomplete("Test prompt", temperature=0.9, max_tokens=100)
    assert result.text == "Custom params response."


def _make_invalid_json_stream_event() -> list:
    """Create one invalid JSON stream event."""
    event = type("Event", (), {})()
    event.data = "{not-valid-json"
    return [event]


def _make_mixed_stream_events_for_provider(provider: str) -> list:
    """Create a valid-invalid-valid stream sequence."""
    if provider == "CohereProvider":
        valid_prefix = _make_cohere_stream_events()[:1]
        valid_suffix = _make_cohere_stream_events()[1:]
    else:
        valid_prefix = _make_meta_stream_events()[:1]
        valid_suffix = _make_meta_stream_events()[1:]
    return valid_prefix + _make_invalid_json_stream_event() + valid_suffix


@pytest.mark.parametrize(
    "test_model_id",
    ["cohere.command-r-16k", "meta.llama-3-70b-instruct"],
)
def test_llm_stream_chat_handles_invalid_json_event(
    monkeypatch: MonkeyPatch, test_model_id: str
) -> None:
    """Ensure stream_chat yields and continues when one event is invalid JSON."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)
    provider = llm._provider.__class__.__name__

    def mocked_chat(*args, **kwargs):
        mock_data = MagicMock()
        mock_data.events = lambda: _make_mixed_stream_events_for_provider(provider)
        mock_response = MagicMock()
        mock_response.data = mock_data
        return mock_response

    monkeypatch.setattr(llm._client, "chat", mocked_chat)

    messages = [ChatMessage(role="user", content="User message")]
    chunks = list(llm.stream_chat(messages))

    assert len(chunks) >= 2
    # Invalid JSON chunk should not break stream and should have empty delta.
    assert any((chunk.delta or "") == "" for chunk in chunks)
    final_content = chunks[-1].message.content or ""
    assert "Stream" in final_content and "response" in final_content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_model_id",
    ["cohere.command-r-16k", "meta.llama-3-70b-instruct"],
)
async def test_llm_astream_chat_handles_invalid_json_event(
    monkeypatch: MonkeyPatch, test_model_id: str
) -> None:
    """Ensure astream_chat yields and continues when one event is invalid JSON."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)
    provider = llm._provider.__class__.__name__

    def mocked_chat(*args, **kwargs):
        mock_data = MagicMock()
        mock_data.events = lambda: _make_mixed_stream_events_for_provider(provider)
        mock_response = MagicMock()
        mock_response.data = mock_data
        return mock_response

    monkeypatch.setattr(llm._client, "chat", mocked_chat)

    messages = [ChatMessage(role="user", content="User message")]
    chunks = []
    astream = llm.astream_chat(messages)
    agen = await astream if hasattr(astream, "__await__") else astream
    async for chunk in agen:
        chunks.append(chunk)

    assert len(chunks) >= 2
    assert any((chunk.delta or "") == "" for chunk in chunks)
    final_content = chunks[-1].message.content or ""
    assert "Stream" in final_content and "response" in final_content


def _make_cohere_tool_stream_events() -> list:
    """Create Cohere-like streaming events with incremental tool calls."""
    events_data = [
        {
            "text": "Calling ",
            "toolCalls": [
                {"name": "search_tool", "parameters": {"q": "llamaindex"}},
            ],
        },
        {
            "text": "tool",
            "toolCalls": [
                {"name": "search_tool", "parameters": {"q": "llamaindex docs"}},
            ],
            "finishReason": "tool_calls",
        },
    ]
    result = []
    for data in events_data:
        event = type("Event", (), {})()
        event.data = json.dumps(data)
        result.append(event)
    return result


@pytest.mark.parametrize("test_model_id", ["cohere.command-r-16k"])
def test_llm_stream_chat_accumulates_tool_calls(
    monkeypatch: MonkeyPatch, test_model_id: str
) -> None:
    """Ensure stream_chat accumulates and merges tool calls by tool name."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    def mocked_chat(*args, **kwargs):
        mock_data = MagicMock()
        mock_data.events = _make_cohere_tool_stream_events
        mock_response = MagicMock()
        mock_response.data = mock_data
        return mock_response

    monkeypatch.setattr(llm._client, "chat", mocked_chat)

    messages = [ChatMessage(role="user", content="Find docs")]
    chunks = list(llm.stream_chat(messages))
    assert len(chunks) == 2

    final_kwargs = chunks[-1].message.additional_kwargs
    assert "tool_calls" in final_kwargs
    assert len(final_kwargs["tool_calls"]) == 1
    assert final_kwargs["tool_calls"][0]["name"] == "search_tool"
    # Latest parameters should win due to merge/update behavior.
    assert "llamaindex docs" in final_kwargs["tool_calls"][0]["input"]


@pytest.mark.asyncio
@pytest.mark.parametrize("test_model_id", ["cohere.command-r-16k"])
async def test_llm_astream_chat_accumulates_tool_calls(
    monkeypatch: MonkeyPatch, test_model_id: str
) -> None:
    """Ensure astream_chat accumulates and merges tool calls by tool name."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    def mocked_chat(*args, **kwargs):
        mock_data = MagicMock()
        mock_data.events = _make_cohere_tool_stream_events
        mock_response = MagicMock()
        mock_response.data = mock_data
        return mock_response

    monkeypatch.setattr(llm._client, "chat", mocked_chat)

    messages = [ChatMessage(role="user", content="Find docs")]
    chunks = []
    astream = llm.astream_chat(messages)
    agen = await astream if hasattr(astream, "__await__") else astream
    async for chunk in agen:
        chunks.append(chunk)

    assert len(chunks) == 2
    final_kwargs = chunks[-1].message.additional_kwargs
    assert "tool_calls" in final_kwargs
    assert len(final_kwargs["tool_calls"]) == 1
    assert final_kwargs["tool_calls"][0]["name"] == "search_tool"
    assert "llamaindex docs" in final_kwargs["tool_calls"][0]["input"]


@pytest.mark.parametrize("test_model_id", ["cohere.command-r-16k"])
def test_llm_stream_complete_propagates_tool_calls(
    monkeypatch: MonkeyPatch, test_model_id: str
) -> None:
    """Ensure stream_complete keeps tool call metadata via conversion helpers."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    def mocked_chat(*args, **kwargs):
        mock_data = MagicMock()
        mock_data.events = _make_cohere_tool_stream_events
        mock_response = MagicMock()
        mock_response.data = mock_data
        return mock_response

    monkeypatch.setattr(llm._client, "chat", mocked_chat)

    chunks = list(llm.stream_complete("Find docs"))
    assert len(chunks) == 2
    final_kwargs = chunks[-1].additional_kwargs
    assert "tool_calls" in final_kwargs
    assert len(final_kwargs["tool_calls"]) == 1
    assert final_kwargs["tool_calls"][0]["name"] == "search_tool"


@pytest.mark.asyncio
@pytest.mark.parametrize("test_model_id", ["cohere.command-r-16k"])
async def test_llm_astream_complete_propagates_tool_calls(
    monkeypatch: MonkeyPatch, test_model_id: str
) -> None:
    """Ensure astream_complete keeps tool call metadata via conversion helpers."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    def mocked_chat(*args, **kwargs):
        mock_data = MagicMock()
        mock_data.events = _make_cohere_tool_stream_events
        mock_response = MagicMock()
        mock_response.data = mock_data
        return mock_response

    monkeypatch.setattr(llm._client, "chat", mocked_chat)

    chunks = []
    astream = llm.astream_complete("Find docs")
    agen = await astream if hasattr(astream, "__await__") else astream
    async for chunk in agen:
        chunks.append(chunk)

    assert len(chunks) == 2
    final_kwargs = chunks[-1].additional_kwargs
    assert "tool_calls" in final_kwargs
    assert len(final_kwargs["tool_calls"]) == 1
    assert final_kwargs["tool_calls"][0]["name"] == "search_tool"


@pytest.mark.parametrize(
    "test_model_id",
    ["cohere.command-r-16k", "meta.llama-3-70b-instruct"],
)
def test_llm_stream_complete_handles_invalid_json_event(
    monkeypatch: MonkeyPatch, test_model_id: str
) -> None:
    """Ensure stream_complete continues when one stream event is invalid JSON."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)
    provider = llm._provider.__class__.__name__

    def mocked_chat(*args, **kwargs):
        mock_data = MagicMock()
        mock_data.events = lambda: _make_mixed_stream_events_for_provider(provider)
        mock_response = MagicMock()
        mock_response.data = mock_data
        return mock_response

    monkeypatch.setattr(llm._client, "chat", mocked_chat)

    chunks = list(llm.stream_complete("hello"))
    assert len(chunks) >= 2
    assert any((chunk.delta or "") == "" for chunk in chunks)
    final_text = chunks[-1].text or ""
    assert "Stream" in final_text and "response" in final_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_model_id",
    ["cohere.command-r-16k", "meta.llama-3-70b-instruct"],
)
async def test_llm_astream_complete_handles_invalid_json_event(
    monkeypatch: MonkeyPatch, test_model_id: str
) -> None:
    """Ensure astream_complete continues when one stream event is invalid JSON."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)
    provider = llm._provider.__class__.__name__

    def mocked_chat(*args, **kwargs):
        mock_data = MagicMock()
        mock_data.events = lambda: _make_mixed_stream_events_for_provider(provider)
        mock_response = MagicMock()
        mock_response.data = mock_data
        return mock_response

    monkeypatch.setattr(llm._client, "chat", mocked_chat)

    chunks = []
    astream = llm.astream_complete("hello")
    agen = await astream if hasattr(astream, "__await__") else astream
    async for chunk in agen:
        chunks.append(chunk)

    assert len(chunks) >= 2
    assert any((chunk.delta or "") == "" for chunk in chunks)
    final_text = chunks[-1].text or ""
    assert "Stream" in final_text and "response" in final_text


def _make_cohere_function_calls_events() -> list:
    """Create stream events using functionCalls alias for tool events."""
    events_data = [
        {
            "text": "Use ",
            "functionCalls": [
                {"name": "search_tool", "parameters": {"q": "alpha"}},
            ],
        },
        {
            "text": "alias",
            "functionCalls": [
                {"name": "search_tool", "parameters": {"q": "beta"}},
            ],
            "finishReason": "tool_calls",
        },
    ]
    result = []
    for data in events_data:
        event = type("Event", (), {})()
        event.data = json.dumps(data)
        result.append(event)
    return result


@pytest.mark.parametrize("test_model_id", ["cohere.command-r-16k"])
def test_llm_stream_chat_function_calls_alias(
    monkeypatch: MonkeyPatch, test_model_id: str
) -> None:
    """Ensure stream_chat supports functionCalls key alias for tool calls."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    def mocked_chat(*args, **kwargs):
        mock_data = MagicMock()
        mock_data.events = _make_cohere_function_calls_events
        mock_response = MagicMock()
        mock_response.data = mock_data
        return mock_response

    monkeypatch.setattr(llm._client, "chat", mocked_chat)

    messages = [ChatMessage(role="user", content="Find docs")]
    chunks = list(llm.stream_chat(messages))
    assert len(chunks) == 2
    final_kwargs = chunks[-1].message.additional_kwargs
    assert "tool_calls" in final_kwargs
    assert final_kwargs["tool_calls"][0]["name"] == "search_tool"
    assert "beta" in final_kwargs["tool_calls"][0]["input"]
