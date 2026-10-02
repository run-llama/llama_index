import asyncio
import json
from typing import Any, AsyncGenerator

import pytest
from pydantic import ValidationError
from ag_ui.core import PROTOCOL_VERSION, RunAgentInput, ToolMessage, UserMessage
from ag_ui.core.types import FileSource, ImagePart, AudioPart, VideoPart, DocumentPart
from llama_index.core.llms.mock import MockLLM
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, LLMMetadata
from llama_index.protocols.ag_ui.router import (
    AGUIWorkflowRouter,
    get_default_workflow_factory,
)
from llama_index.protocols.ag_ui.utils import (
    ag_ui_message_to_llama_index_message,
    llama_index_message_to_ag_ui_message,
)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (" plain text ", " plain text "),
        ([], ""),
        (
            [{"type": "text", "text": " first "}, {"type": "text", "text": "second\n"}],
            " first second\n",
        ),
    ],
)
def test_tool_content_is_text(content, expected) -> None:
    message = ToolMessage(
        id="result", role="tool", tool_call_id="call", content=content
    )
    converted = ag_ui_message_to_llama_index_message(message)
    assert converted.content == expected
    assert converted.additional_kwargs == {"id": "result", "tool_call_id": "call"}
    back = llama_index_message_to_ag_ui_message(converted)
    assert back.content == expected
    assert back.tool_call_id == "call"


@pytest.mark.parametrize("part_type", ["image", "audio", "video", "document"])
@pytest.mark.parametrize(
    "source",
    [
        {"type": "url", "value": "https://example.com/media"},
        {"type": "data", "value": "dGVzdA==", "mimeType": "application/octet-stream"},
        {"type": "file", "value": "opaque-secret-handle", "provider": "openai"},
    ],
)
def test_tool_media_is_skipped_with_warning(part_type, source, caplog) -> None:
    message = ToolMessage.model_validate(
        {
            "id": "result",
            "role": "tool",
            "toolCallId": "call",
            "content": [
                {"type": "text", "text": "before "},
                {"type": part_type, "source": source},
                {"type": "text", "text": "after"},
            ],
        }
    )
    converted = ag_ui_message_to_llama_index_message(message)
    assert converted.content == "before after"
    assert "tool result" in caplog.text
    assert part_type in caplog.text
    assert source["value"] not in caplog.text


@pytest.mark.parametrize("part_class", [ImagePart, AudioPart, VideoPart, DocumentPart])
def test_provider_files_are_explicitly_skipped(part_class, caplog) -> None:
    message = UserMessage(
        id="user",
        role="user",
        content=[
            part_class(
                source=FileSource(
                    value="https://opaque-provider-handle", provider="openai"
                )
            )
        ],
    )
    converted = ag_ui_message_to_llama_index_message(message)
    assert converted.blocks == []
    assert "provider file" in caplog.text
    assert "https://opaque-provider-handle" not in caplog.text


class TextResponseLLM(MockLLM, FunctionCallingLLM):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True, is_chat_model=True)

    def _prepare_chat_with_tools(self, tools, **kwargs) -> dict[str, Any]:
        return kwargs

    # Use APIs available at the package's minimum llama-index-core version.
    async def astream_chat_with_tools(
        self, **kwargs
    ) -> AsyncGenerator[ChatResponse, None]:
        async def responses() -> AsyncGenerator[ChatResponse, None]:
            yield ChatResponse(
                message=ChatMessage(role="assistant", content="ok"), delta="ok"
            )

        return responses()

    def get_tool_calls_from_response(self, response, **kwargs) -> list:
        return []


def test_router_accepts_structured_tool_results_and_declares_protocol_version() -> None:
    router = AGUIWorkflowRouter(get_default_workflow_factory(llm=TextResponseLLM()))
    request = RunAgentInput.model_validate(
        {
            "threadId": "thread",
            "runId": "run",
            "state": {},
            "tools": [],
            "context": [],
            "forwardedProps": {},
            "messages": [
                {
                    "id": "result",
                    "role": "tool",
                    "toolCallId": "call",
                    "content": [{"type": "text", "text": "done"}],
                }
            ],
        }
    )

    async def collect() -> list[dict[str, Any]]:
        response = await router.run(request)
        return [
            json.loads(chunk.removeprefix("data: ").strip())
            async for chunk in response.body_iterator
        ]

    events = asyncio.run(collect())
    assert events[0]["type"] == "RUN_STARTED"
    assert events[0]["protocolVersion"] == PROTOCOL_VERSION
    assert events[-1]["type"] == "RUN_FINISHED"
    assert not any(event["type"] == "RUN_ERROR" for event in events)
    snapshot = next(event for event in events if event["type"] == "MESSAGES_SNAPSHOT")
    result = next(
        message for message in snapshot["messages"] if message["id"] == "result"
    )
    assert result["content"] == "done"
    assert result["toolCallId"] == "call"


def test_binary_wire_content_requires_migration() -> None:
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        UserMessage.model_validate(
            {
                "id": "old",
                "role": "user",
                "content": [
                    {"type": "binary", "mimeType": "image/png", "data": "dGVzdA=="}
                ],
            }
        )


def test_media_only_tool_result_becomes_empty_text(caplog) -> None:
    message = ToolMessage.model_validate(
        {
            "id": "result",
            "role": "tool",
            "toolCallId": "call",
            "content": [
                {"type": "image", "source": {"type": "file", "value": "file-123"}}
            ],
        }
    )
    converted = ag_ui_message_to_llama_index_message(message)
    assert converted.content == ""
    assert llama_index_message_to_ag_ui_message(converted).content == ""
    assert "tool result" in caplog.text
