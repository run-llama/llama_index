# Test OCI Generative AI LLM service

from unittest.mock import MagicMock
from typing import Any

import pytest
from pytest import MonkeyPatch

from llama_index.llms.oci_genai import OCIGenAI
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
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
        elif provider == "MetaProvider":
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
        elif provider == "MetaProvider":
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
    if provider == "MetaProvider":
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


@pytest.mark.parametrize(
    "test_model_id", ["cohere.command-r-16k", "cohere.command-r-plus"]
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
        tool_calls = [
            MockResponseDict(
                {
                    "name": "mock_tool_function",
                    "parameters": {"param1": "test"},
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
    expected_tool_calls = [
        {
            "name": "mock_tool_function",
            "toolUseId": actual_response.message.additional_kwargs["tool_calls"][0][
                "toolUseId"
            ],
            "input": json.dumps({"param1": "test"}),
        }
    ]

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
