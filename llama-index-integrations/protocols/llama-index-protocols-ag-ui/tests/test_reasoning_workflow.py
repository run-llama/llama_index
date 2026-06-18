"""
End-to-end test that ``AGUIChatWorkflow`` emits AG-UI ``REASONING_MESSAGE_*``
events when the upstream LLM streams a reasoning channel.

Driven by a stub ``FunctionCallingLLM`` whose ``astream_chat_with_tools`` yields
``ChatResponse`` chunks shaped like the OpenAI Responses-API stream (raw
``response.reasoning_summary_text.delta`` events followed by plain text
deltas). This is exactly the shape ``OpenAIResponses`` exposes through
``ChatResponse.raw``; running it against a stub avoids any network or model
key dependency.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, AsyncIterator, List, Sequence

import pytest

from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.protocols.ag_ui.agent import (
    AGUIChatWorkflow,
    _RESPONSES_REASONING_DELTA_TYPE,
)
from llama_index.protocols.ag_ui.events import (
    ReasoningMessageContentWorkflowEvent,
    ReasoningMessageEndWorkflowEvent,
    ReasoningMessageStartWorkflowEvent,
    TextMessageChunkWorkflowEvent,
)


class _ReasoningStubLLM(FunctionCallingLLM):
    """
    Minimal FunctionCallingLLM that streams a synthetic reasoning+text chunk
    sequence so the workflow has something to translate.
    """

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name="stub-reasoning",
            is_function_calling_model=True,
            is_chat_model=True,
        )

    async def astream_chat_with_tools(
        self,
        tools: Sequence[Any],
        chat_history: List[ChatMessage],
        allow_parallel_tool_calls: bool = True,
        **kwargs: Any,
    ) -> AsyncIterator[ChatResponse]:
        async def _gen() -> AsyncIterator[ChatResponse]:
            # Two reasoning deltas, then two text deltas. The workflow should
            # emit Start -> Content x2 -> End -> Text x2.
            for r in ("step ", "one. "):
                yield ChatResponse(
                    message=ChatMessage(role="assistant", content=""),
                    delta="",
                    raw=SimpleNamespace(
                        type=_RESPONSES_REASONING_DELTA_TYPE,
                        delta=r,
                    ),
                )
            for t in ("hello ", "world"):
                yield ChatResponse(
                    message=ChatMessage(role="assistant", content=t),
                    delta=t,
                    raw=SimpleNamespace(type="response.output_text.delta", delta=t),
                )

        return _gen()

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[Any]:
        return []

    # The remaining abstract methods aren't reached by this test, but the
    # FunctionCallingLLM ABC requires concrete bodies.
    def chat(self, *args: Any, **kwargs: Any) -> ChatResponse:  # pragma: no cover
        raise NotImplementedError

    def stream_chat(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError

    def complete(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError

    def stream_complete(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError

    async def achat(
        self, *args: Any, **kwargs: Any
    ) -> ChatResponse:  # pragma: no cover
        raise NotImplementedError

    async def acomplete(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError

    async def astream_complete(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError

    async def astream_chat(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError

    def _prepare_chat_with_tools(
        self, *args: Any, **kwargs: Any
    ) -> dict:  # pragma: no cover
        return {}


@pytest.mark.asyncio
async def test_workflow_emits_reasoning_then_text() -> None:
    from ag_ui.core import RunAgentInput, UserMessage

    workflow = AGUIChatWorkflow(
        llm=_ReasoningStubLLM(),
        frontend_tools=[],
        backend_tools=[],
        initial_state={},
        system_prompt=None,
        timeout=30,
    )

    run_input = RunAgentInput(
        thread_id="t1",
        run_id="r1",
        messages=[UserMessage(id="m1", role="user", content="hi")],
        tools=[],
        context=[],
        state={},
        forwarded_props={},
    )

    handler = workflow.run(input_data=run_input)

    seen: List[Any] = []
    async for ev in handler.stream_events():
        seen.append(ev)
    await handler

    reasoning_start = [
        e for e in seen if isinstance(e, ReasoningMessageStartWorkflowEvent)
    ]
    reasoning_content = [
        e for e in seen if isinstance(e, ReasoningMessageContentWorkflowEvent)
    ]
    reasoning_end = [e for e in seen if isinstance(e, ReasoningMessageEndWorkflowEvent)]
    text_chunks = [e for e in seen if isinstance(e, TextMessageChunkWorkflowEvent)]

    assert len(reasoning_start) == 1, "exactly one REASONING_MESSAGE_START"
    assert len(reasoning_end) == 1, "exactly one REASONING_MESSAGE_END"
    assert len(reasoning_content) == 2, "two reasoning content deltas"
    assert "".join(e.delta for e in reasoning_content) == "step one. "
    assert len(text_chunks) == 2, "two text chunks after reasoning closes"
    assert "".join(e.delta for e in text_chunks) == "hello world"

    # Ordering: REASONING_MESSAGE_END must come before the first
    # TEXT_MESSAGE_CHUNK so the reasoning slot finalizes ahead of the
    # assistant text in the AG-UI consumer.
    reasoning_end_idx = seen.index(reasoning_end[0])
    first_text_idx = seen.index(text_chunks[0])
    assert reasoning_end_idx < first_text_idx
