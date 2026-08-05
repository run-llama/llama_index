"""OpenTelemetry GenAI semantic-convention event handling for LlamaIndex."""

from __future__ import annotations

from typing import Any, Union

from llama_index.core.instrumentation.events.embedding import (
    EmbeddingEndEvent,
    EmbeddingStartEvent,
)
from llama_index.core.instrumentation.events.agent import (
    AgentChatWithStepEndEvent,
    AgentChatWithStepStartEvent,
    AgentRunStepEndEvent,
    AgentRunStepStartEvent,
    AgentToolCallEvent,
    AgentToolCallEndEvent,
    AgentToolCallStartEvent,
)
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
)
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalEndEvent,
    RetrievalStartEvent,
)
from llama_index_instrumentation.base.event import BaseEvent
from llama_index_instrumentation.event_handlers import BaseEventHandler
from opentelemetry.trace import TracerProvider
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    EmbeddingInvocation,
    InferenceInvocation,
    RetrievalInvocation,
    ToolInvocation,
)
from opentelemetry.util.genai.types import InputMessage, OutputMessage, Text
from pydantic import PrivateAttr


_Invocation = Union[
    AgentInvocation,
    EmbeddingInvocation,
    InferenceInvocation,
    RetrievalInvocation,
    ToolInvocation,
]


def _model_value(model_dict: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = model_dict.get(name)
        if value is not None:
            return str(value)
    return None


def _provider_and_model(model_dict: dict[str, Any]) -> tuple[str, str | None]:
    provider = _model_value(model_dict, "provider", "model_provider")
    model = _model_value(model_dict, "model_name", "model", "model_id")
    return provider or "llama_index", model


def _usage_value(response: Any, *names: str) -> int | None:
    metadata = getattr(response, "additional_kwargs", {}) or {}
    raw = getattr(response, "raw", None)
    if isinstance(raw, dict):
        metadata = {**raw, **metadata}
    usage = metadata.get("usage", metadata)
    if not isinstance(usage, dict):
        return None
    for name in names:
        value = usage.get(name)
        if isinstance(value, int):
            return value
    return None


def _query_text(query: Any) -> str:
    return query if isinstance(query, str) else str(getattr(query, "query_str", query))


def _retrieval_documents(nodes: list[Any]) -> list[dict[str, Any]]:
    """Convert LlamaIndex retrieval results to the GenAI document representation."""
    documents = []
    for node_with_score in nodes:
        node = node_with_score.node
        documents.append(
            {
                "id": node.node_id,
                "content": node.get_content(),
                "score": node_with_score.score,
            }
        )
    return documents


class GenAIEventHandler(BaseEventHandler):
    """
    Translate LlamaIndex events into OpenTelemetry GenAI telemetry.

    The handler creates GenAI invocations for LLM chat/completion and embedding
    events. It supports LLM, embedding, retrieval, tool, and agent operations.
    It is registered only when ``LlamaIndexOpenTelemetry`` is configured with
    ``genai_enabled=True``.

    Args:
        tracer_provider (TracerProvider): The provider used for generated GenAI
            spans and metrics.

    """

    _telemetry: TelemetryHandler = PrivateAttr()
    _active: dict[str, _Invocation] = PrivateAttr(default_factory=dict)

    def __init__(self, tracer_provider: TracerProvider) -> None:
        """
        Initialize the GenAI event handler.

        Args:
            tracer_provider (TracerProvider): The provider shared with the
                generic LlamaIndex OpenTelemetry integration.

        """
        super().__init__()
        self._telemetry = TelemetryHandler(tracer_provider=tracer_provider)

    def handle(self, event: BaseEvent, **_: Any) -> None:
        """Process a LlamaIndex event and complete its matching invocation."""
        if isinstance(event, AgentToolCallEvent):
            self._record_tool_call(event)
            return
        if isinstance(event, AgentToolCallStartEvent):
            self._start_tool_call(event)
            return
        if isinstance(event, AgentToolCallEndEvent):
            self._finish_tool_call(event)
            return

        key = event.span_id
        if key is None:
            return

        if isinstance(event, LLMChatStartEvent):
            provider, model = _provider_and_model(event.model_dict)
            invocation = self._telemetry.inference(
                provider, request_model=model, operation_name="chat"
            )
            self._set_chat_input(invocation, event.messages)
            self._set_request_parameters(invocation, event.additional_kwargs)
            self._active[key] = invocation
        elif isinstance(event, LLMCompletionStartEvent):
            provider, model = _provider_and_model(event.model_dict)
            invocation = self._telemetry.inference(
                provider, request_model=model, operation_name="completion"
            )
            if self._telemetry.should_capture_content():
                invocation.input_messages = [
                    InputMessage(role="user", parts=[Text(content=event.prompt)])
                ]
            self._set_request_parameters(invocation, event.additional_kwargs)
            self._active[key] = invocation
        elif isinstance(event, EmbeddingStartEvent):
            provider, model = _provider_and_model(event.model_dict)
            self._active[key] = self._telemetry.embedding(provider, request_model=model)
        elif isinstance(event, LLMChatEndEvent):
            invocation = self._pop(key, InferenceInvocation)
            if invocation is not None:
                self._finish_chat(invocation, event)
        elif isinstance(event, LLMCompletionEndEvent):
            invocation = self._pop(key, InferenceInvocation)
            if invocation is not None:
                self._finish_completion(invocation, event)
        elif isinstance(event, EmbeddingEndEvent):
            invocation = self._pop(key, EmbeddingInvocation)
            if invocation is not None:
                if event.embeddings:
                    invocation.dimension_count = len(event.embeddings[0])
                invocation.stop()
        elif isinstance(event, RetrievalStartEvent):
            invocation = self._telemetry.retrieval()
            invocation.query_text = _query_text(event.str_or_query_bundle)
            self._active[key] = invocation
        elif isinstance(event, RetrievalEndEvent):
            invocation = self._pop(key, RetrievalInvocation)
            if invocation is not None:
                invocation.documents = _retrieval_documents(event.nodes)
                invocation.stop()
        elif isinstance(event, AgentChatWithStepStartEvent):
            invocation = self._telemetry.invoke_local_agent()
            self._set_agent_input(invocation, event.user_msg)
            self._active[key] = invocation
        elif isinstance(event, AgentChatWithStepEndEvent):
            invocation = self._pop(key, AgentInvocation)
            if invocation is not None:
                self._finish_agent(invocation, event.response)
        elif isinstance(event, AgentRunStepStartEvent):
            invocation = self._telemetry.invoke_local_agent()
            invocation.agent_id = event.task_id
            self._set_agent_input(invocation, event.input)
            self._active[key] = invocation
        elif isinstance(event, AgentRunStepEndEvent):
            invocation = self._pop(key, AgentInvocation)
            if invocation is not None:
                self._finish_agent(invocation, event.step_output)

    def _pop(self, key: str, expected_type: type[_Invocation]) -> _Invocation | None:
        invocation = self._active.pop(key, None)
        return invocation if isinstance(invocation, expected_type) else None

    def _set_chat_input(
        self, invocation: InferenceInvocation, messages: list[Any]
    ) -> None:
        if not self._telemetry.should_capture_content():
            return
        invocation.input_messages = [
            InputMessage(
                role=str(
                    message.role.value
                    if hasattr(message.role, "value")
                    else message.role
                ),
                parts=[Text(content=message.content or "")],
            )
            for message in messages
        ]

    @staticmethod
    def _set_request_parameters(
        invocation: InferenceInvocation, kwargs: dict[str, Any]
    ) -> None:
        for field in ("temperature", "top_p", "max_tokens", "seed"):
            value = kwargs.get(field)
            if value is not None:
                setattr(invocation, field, value)

    def _finish_chat(
        self, invocation: InferenceInvocation, event: LLMChatEndEvent
    ) -> None:
        response = event.response
        if response is not None:
            if self._telemetry.should_capture_content():
                invocation.output_messages = [
                    OutputMessage(
                        role=str(response.message.role.value),
                        parts=[Text(content=response.message.content or "")],
                        finish_reason=str(
                            response.additional_kwargs.get("finish_reason", "stop")
                        ),
                    )
                ]
            invocation.input_tokens = _usage_value(
                response, "prompt_tokens", "input_tokens"
            )
            invocation.output_tokens = _usage_value(
                response, "completion_tokens", "output_tokens"
            )
            invocation.response_model_name = _model_value(
                response.additional_kwargs, "model", "model_name"
            )
        invocation.stop()

    def _finish_completion(
        self, invocation: InferenceInvocation, event: LLMCompletionEndEvent
    ) -> None:
        response = event.response
        if self._telemetry.should_capture_content():
            invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content=response.text)],
                    finish_reason=str(
                        response.additional_kwargs.get("finish_reason", "stop")
                    ),
                )
            ]
        invocation.input_tokens = _usage_value(
            response, "prompt_tokens", "input_tokens"
        )
        invocation.output_tokens = _usage_value(
            response, "completion_tokens", "output_tokens"
        )
        invocation.response_model_name = _model_value(
            response.additional_kwargs, "model", "model_name"
        )
        invocation.stop()

    def _record_tool_call(self, event: AgentToolCallEvent) -> None:
        """Record the currently single-event LlamaIndex tool lifecycle."""
        tool = event.tool
        invocation: ToolInvocation = self._telemetry.tool(
            tool.name or "unknown",
            tool_type="function",
            tool_description=tool.description,
        )
        if self._telemetry.should_capture_content():
            invocation.arguments = event.arguments
        invocation.stop()

    @staticmethod
    def _tool_key(tool_id: str) -> str:
        return f"tool:{tool_id}"

    def _start_tool_call(self, event: AgentToolCallStartEvent) -> None:
        invocation: ToolInvocation = self._telemetry.tool(
            event.tool_name,
            tool_call_id=event.tool_id,
            tool_type="function",
            tool_description=event.tool_description,
        )
        if self._telemetry.should_capture_content():
            invocation.arguments = event.tool_kwargs
        self._active[self._tool_key(event.tool_id)] = invocation

    def _finish_tool_call(self, event: AgentToolCallEndEvent) -> None:
        invocation = self._pop(self._tool_key(event.tool_id), ToolInvocation)
        if invocation is None:
            return
        output = event.tool_output
        if self._telemetry.should_capture_content():
            invocation.tool_result = output.content
        if output.is_error:
            invocation.fail(
                output.exception
                or RuntimeError(output.content or f"Tool {event.tool_name} failed")
            )
            return
        invocation.stop()

    def _set_agent_input(self, invocation: AgentInvocation, value: Any) -> None:
        if self._telemetry.should_capture_content() and value is not None:
            invocation.input_messages = [
                InputMessage(role="user", parts=[Text(content=str(value))])
            ]

    def _finish_agent(self, invocation: AgentInvocation, output: Any) -> None:
        if self._telemetry.should_capture_content() and output is not None:
            response = getattr(output, "response", output)
            invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content=str(response))],
                    finish_reason="stop",
                )
            ]
        invocation.stop()
