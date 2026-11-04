"""
Tests for FunctionAgent recovering a final answer that a model placed in its
reasoning channel (reasoning_content / ThinkingBlock) instead of the normal text
content.

Regression coverage for https://github.com/run-llama/llama_index/issues/21337:
some OpenAI-compatible models (e.g. Kimi-K2.5) occasionally return the final
answer in ``reasoning_content`` while ``content`` is empty. Without recovery,
FunctionAgent silently returns an empty answer (unlike ReActAgent, which
validates for empty content).
"""

from typing import Any, AsyncGenerator, List, Optional

import pytest

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow.function_agent import FunctionAgent as FA
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    LLMMetadata,
    TextBlock,
    ThinkingBlock,
)
from llama_index.core.llms import MockLLM

FINAL_ANSWER = "The capital of France is Paris."


class ScriptedFunctionLLM(MockLLM):
    """
    A function-calling LLM whose single final response is fully controllable.

    ``response`` is emitted verbatim on both the streaming and non-streaming
    tool-calling entry points so the same scenario can be exercised on both
    FunctionAgent code paths.
    """

    _response: ChatResponse

    def __init__(self, response: ChatResponse) -> None:
        super().__init__()
        self._response = response

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)

    async def astream_chat_with_tools(
        self,
        tools: List[Any],
        chat_history: Optional[List[ChatMessage]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatResponse, None]:
        response = self._response

        async def _gen() -> AsyncGenerator[ChatResponse, None]:
            yield response

        return _gen()

    async def achat_with_tools(
        self,
        tools: List[Any],
        user_msg: Any = None,
        chat_history: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        return self._response

    def get_tool_calls_from_response(
        self, response: ChatResponse, error_on_no_tool_call: bool = False, **kwargs: Any
    ) -> List[Any]:
        # These scenarios never emit tool calls; the final answer is terminal.
        return []


def _reasoning_only_response() -> ChatResponse:
    """Final response with empty text content, answer only in a ThinkingBlock."""
    return ChatResponse(
        message=ChatMessage(
            role="assistant",
            blocks=[ThinkingBlock(content=FINAL_ANSWER)],
        ),
        delta="",
        raw={
            "choices": [
                {"message": {"content": None, "reasoning_content": FINAL_ANSWER}}
            ]
        },
    )


def _reasoning_only_response_raw_only() -> ChatResponse:
    """Answer only in the raw provider payload's reasoning_content (no ThinkingBlock)."""
    return ChatResponse(
        message=ChatMessage(role="assistant", content=""),
        delta="",
        raw={
            "choices": [
                {"message": {"content": None, "reasoning_content": FINAL_ANSWER}}
            ]
        },
    )


def _normal_response() -> ChatResponse:
    """Well-behaved final response: answer in content, a ThinkingBlock also present."""
    return ChatResponse(
        message=ChatMessage(
            role="assistant",
            blocks=[
                ThinkingBlock(content="internal reasoning, not the answer"),
                TextBlock(text=FINAL_ANSWER),
            ],
        ),
        delta=FINAL_ANSWER,
        raw={"choices": [{"message": {"content": FINAL_ANSWER}}]},
    )


def _empty_response() -> ChatResponse:
    """No tool calls, no content, no reasoning content at all."""
    return ChatResponse(
        message=ChatMessage(role="assistant", content=""),
        delta="",
        raw={"choices": [{"message": {"content": None}}]},
    )


async def _run(agent: FunctionAgent) -> str:
    handler = agent.run(user_msg="What is the capital of France?")
    result = await handler
    return result.response.content or ""


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_reasoning_only_thinking_block_is_recovered(streaming: bool) -> None:
    """Answer stranded in a ThinkingBlock is promoted to content (both paths)."""
    agent = FunctionAgent(llm=ScriptedFunctionLLM(_reasoning_only_response()), tools=[])
    agent.streaming = streaming

    assert await _run(agent) == FINAL_ANSWER


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_reasoning_only_raw_payload_is_recovered(streaming: bool) -> None:
    """Answer only in raw reasoning_content (no ThinkingBlock) is recovered too."""
    agent = FunctionAgent(
        llm=ScriptedFunctionLLM(_reasoning_only_response_raw_only()), tools=[]
    )
    agent.streaming = streaming

    assert await _run(agent) == FINAL_ANSWER


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_present_content_is_not_overridden_by_thinking(streaming: bool) -> None:
    """
    Positive guard: when content is present it must be returned as-is, never
    replaced by reasoning content.
    """
    agent = FunctionAgent(llm=ScriptedFunctionLLM(_normal_response()), tools=[])
    agent.streaming = streaming

    assert await _run(agent) == FINAL_ANSWER


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_truly_empty_response_stays_empty(streaming: bool) -> None:
    """
    No content and no reasoning content: behavior is unchanged (empty),
    the fix does not invent an answer.
    """
    agent = FunctionAgent(llm=ScriptedFunctionLLM(_empty_response()), tools=[])
    agent.streaming = streaming

    assert await _run(agent) == ""


def test_extract_reasoning_content_sources() -> None:
    """Unit coverage of the recovery helper's ordered source resolution."""
    # 1. ThinkingBlock(s) win, and multiple non-empty blocks are joined.
    multi = ChatResponse(
        message=ChatMessage(
            role="assistant",
            blocks=[ThinkingBlock(content="foo"), ThinkingBlock(content="bar")],
        ),
        raw={},
    )
    assert FA._extract_reasoning_content(multi) == "foobar"

    # 2. Fallback to additional_kwargs reasoning_content.
    kwargs_only = ChatResponse(
        message=ChatMessage(
            role="assistant",
            content="",
            additional_kwargs={"reasoning_content": FINAL_ANSWER},
        ),
        raw={},
    )
    assert FA._extract_reasoning_content(kwargs_only) == FINAL_ANSWER

    # 3. Fallback to raw choices[0].message.reasoning_content.
    raw_only = _reasoning_only_response_raw_only()
    assert FA._extract_reasoning_content(raw_only) == FINAL_ANSWER

    # 4. Nothing available -> None.
    nothing = ChatResponse(message=ChatMessage(role="assistant", content=""), raw={})
    assert FA._extract_reasoning_content(nothing) is None
