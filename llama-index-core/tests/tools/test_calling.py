import pytest
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from unittest.mock import patch

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    CompletionResponseAsyncGen,
)
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools import calling as tools_calling
from llama_index.core.tools.calling import (
    acall_tool_with_selection,
    call_tool_with_selection,
)


def _basic_echo_tool(x: int) -> str:
    return f"echo:{x}"


def _error_tool(x: int) -> str:
    raise ValueError("tool failed")


async def _async_tool(x: int) -> str:
    return f"async:{x}"


class _RecordingToolCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__([], [])
        self.started: List[Tuple[CBEventType, Optional[Dict[str, Any]], str]] = []
        self.ended: List[Tuple[CBEventType, Optional[Dict[str, Any]], str]] = []

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        self.started.append((event_type, payload, event_id))
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        self.ended.append((event_type, payload, event_id))

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(self, *args: Any, **kwargs: Any) -> None:
        pass


class _FunctionCallingConsumer(FunctionCallingLLM):
    def __init__(self, tool_calls: List[ToolSelection]) -> None:
        super().__init__()
        self._tool_calls = tool_calls

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[Any],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {"messages": []}

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        return self._tool_calls

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="assistant", content=""))

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="assistant", content=""))

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return CompletionResponse(text="")

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError


def _assert_tool_payload(
    payload: Optional[Dict[str, Any]],
    *,
    expected_tool_name: str,
    expected_tool_id: str,
    expected_tool_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    assert payload is not None
    assert EventPayload.TOOL in payload
    tool_payload = payload[EventPayload.TOOL]
    assert isinstance(tool_payload, Dict)
    assert tool_payload["tool_name"] == expected_tool_name
    assert tool_payload["tool_kwargs"] == (
        expected_tool_kwargs if expected_tool_kwargs is not None else {"x": 1}
    )
    assert tool_payload["tool_id"] == expected_tool_id
    return tool_payload


def test_call_tool_with_selection_emits_tool_audit_event() -> None:
    tool = FunctionTool.from_defaults(_basic_echo_tool, name="echo")
    selection = ToolSelection(
        tool_id="sync-tool", tool_name="echo", tool_kwargs={"x": 1}
    )
    handler = _RecordingToolCallbackHandler()

    output = call_tool_with_selection(
        tool_call=selection,
        tools=[tool],
        verbose=False,
        callback_manager=CallbackManager([handler]),
        tool_id=selection.tool_id,
    )

    assert output.content == "echo:1"
    assert len(handler.started) == 1
    assert len(handler.ended) == 1
    assert handler.started[0][0] == CBEventType.TOOL
    assert handler.ended[0][0] == CBEventType.TOOL
    assert (
        _assert_tool_payload(
            handler.started[0][1],
            expected_tool_name="echo",
            expected_tool_id="sync-tool",
        )["is_error"]
        is False
    )
    assert (
        _assert_tool_payload(
            handler.ended[0][1],
            expected_tool_name="echo",
            expected_tool_id="sync-tool",
        )["output"]
        == "echo:1"
    )
    assert (
        _assert_tool_payload(
            handler.started[0][1],
            expected_tool_name="echo",
            expected_tool_id="sync-tool",
        )["output"]
        is None
    )


@pytest.mark.asyncio
async def test_acall_tool_with_selection_emits_tool_audit_event() -> None:
    tool = FunctionTool.from_defaults(
        _async_tool, async_fn=_async_tool, name="echo_async"
    )
    selection = ToolSelection(
        tool_id="async-tool", tool_name="echo_async", tool_kwargs={"x": 2}
    )
    handler = _RecordingToolCallbackHandler()

    output = await acall_tool_with_selection(
        tool_call=selection,
        tools=[tool],
        callback_manager=CallbackManager([handler]),
        tool_id=selection.tool_id,
    )

    assert output.content == "async:2"
    assert len(handler.started) == 1
    assert len(handler.ended) == 1
    assert handler.started[0][0] == CBEventType.TOOL
    assert handler.ended[0][0] == CBEventType.TOOL
    assert (
        _assert_tool_payload(
            handler.started[0][1],
            expected_tool_name="echo_async",
            expected_tool_id="async-tool",
            expected_tool_kwargs={"x": 2},
        )["is_error"]
        is False
    )
    assert (
        _assert_tool_payload(
            handler.ended[0][1],
            expected_tool_name="echo_async",
            expected_tool_id="async-tool",
            expected_tool_kwargs={"x": 2},
        )["output"]
        == "async:2"
    )


def test_tool_audit_event_marks_error_outputs() -> None:
    tool = FunctionTool.from_defaults(_error_tool, name="error")
    selection = ToolSelection(
        tool_id="error-tool", tool_name="error", tool_kwargs={"x": 4}
    )
    handler = _RecordingToolCallbackHandler()

    output = call_tool_with_selection(
        tool_call=selection,
        tools=[tool],
        callback_manager=CallbackManager([handler]),
        tool_id=selection.tool_id,
    )

    assert output.is_error is True
    assert output.content == "Encountered error: tool failed"
    end_payload = _assert_tool_payload(
        handler.ended[0][1],
        expected_tool_name="error",
        expected_tool_id="error-tool",
        expected_tool_kwargs={"x": 4},
    )
    assert end_payload["is_error"] is True
    assert end_payload["error"] == "tool failed"


def test_call_tool_with_selection_no_callback_manager_preserves_behavior() -> None:
    tool = FunctionTool.from_defaults(_basic_echo_tool, name="echo")
    selection = ToolSelection(
        tool_id="no-callback-tool", tool_name="echo", tool_kwargs={"x": 1}
    )

    output_without_callback = call_tool_with_selection(
        tool_call=selection,
        tools=[tool],
        verbose=False,
    )
    output_with_callback = call_tool_with_selection(
        tool_call=selection,
        tools=[tool],
        verbose=False,
        callback_manager=CallbackManager([]),
        tool_id=selection.tool_id,
    )

    assert output_without_callback.is_error == output_with_callback.is_error
    assert output_without_callback.content == output_with_callback.content
    assert output_without_callback.tool_name == output_with_callback.tool_name


def test_selected_tool_consumer_passes_callback_manager() -> None:
    tool = FunctionTool.from_defaults(_basic_echo_tool, name="echo")
    selection = ToolSelection(
        tool_id="consumer-tool", tool_name="echo", tool_kwargs={"x": 1}
    )
    llm = _FunctionCallingConsumer([selection])
    llm.callback_manager = CallbackManager([_RecordingToolCallbackHandler()])
    original_call_tool = tools_calling.call_tool_with_selection

    with patch(
        "llama_index.core.tools.calling.call_tool_with_selection"
    ) as mock_call_tool:
        mock_call_tool.side_effect = original_call_tool
        llm.predict_and_call(tools=[tool], error_on_tool_error=True)

        assert mock_call_tool.called
        assert mock_call_tool.call_count == 1
        _, kwargs = mock_call_tool.call_args
        assert kwargs["callback_manager"] is llm.callback_manager
        assert kwargs["tool_id"] == selection.tool_id
