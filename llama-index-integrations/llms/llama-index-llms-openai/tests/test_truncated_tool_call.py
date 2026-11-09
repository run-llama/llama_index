"""
Tier 1: a tool call from a reply the provider marked truncated must not run.

The response objects below are the real OpenAI SDK types, so the payload shape is the
one the client sees at runtime rather than a stand-in.
"""

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ToolCallBlock
from llama_index.llms.openai import OpenAI

TRUNCATED_ARGS = '{"path": "report.txt", "content": "line 01: the quick brown fox'


def _tool_block() -> ToolCallBlock:
    return ToolCallBlock(
        tool_call_id="call_1", tool_name="write_file", tool_kwargs=TRUNCATED_ARGS
    )


def _raw(finish_reason: str | None = None, usage_only: bool = False) -> ChatCompletion:
    """
    A real `ChatCompletion`.

    `usage_only=True` models the final stream chunk when usage reporting is on: the
    chunk carries usage and no choices at all.
    """
    if usage_only:
        return ChatCompletion(
            id="chatcmpl-1",
            choices=[],
            created=0,
            model="gpt-4o-mini",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    return ChatCompletion(
        id="chatcmpl-1",
        choices=[
            Choice(
                index=0,
                finish_reason=finish_reason,
                message=ChatCompletionMessage(role="assistant", content=None),
                logprobs=None,
            )
        ],
        created=0,
        model="gpt-4o-mini",
        object="chat.completion",
    )


def _response(finish_reason: str | None, with_tool_call: bool = True) -> ChatResponse:
    # NOTE: do not pass content="" here. In this version ChatMessage replaces `blocks`
    # with a single empty TextBlock whenever content is given, which would silently drop
    # the tool call and make every assertion below meaningless.
    blocks = [_tool_block()] if with_tool_call else []
    return ChatResponse(
        message=ChatMessage(role="assistant", blocks=blocks),
        raw=_raw(finish_reason),
    )


@pytest.fixture()
def llm() -> OpenAI:
    return OpenAI(model="gpt-4o-mini", api_key="unused")


def test_truncated_tool_call_raises(llm: OpenAI) -> None:
    with pytest.raises(ValueError, match="truncated"):
        llm.get_tool_calls_from_response(_response("length"))


def test_truncated_without_tool_call_no_error(llm: OpenAI) -> None:
    # truncated text with no tool call is out of scope for tier 1: it must not be
    # reported as truncation, only as the ordinary "no tool call" case
    assert (
        llm.get_tool_calls_from_response(
            _response("length", with_tool_call=False), error_on_no_tool_call=False
        )
        == []
    )
    with pytest.raises(ValueError, match="Expected at least one tool call"):
        llm.get_tool_calls_from_response(_response("length", with_tool_call=False))


def test_truncated_with_error_flag_false_raises(llm: OpenAI) -> None:
    # FunctionAgent passes error_on_no_tool_call=False; the guard must not depend on it
    with pytest.raises(ValueError, match="truncated"):
        llm.get_tool_calls_from_response(
            _response("length"), error_on_no_tool_call=False
        )


def test_usage_only_final_chunk_no_crash(llm: OpenAI) -> None:
    # a usage-only final chunk leaves `choices` empty; the guard must not index it
    message = ChatMessage(role="assistant", blocks=[_tool_block()])
    calls = llm.get_tool_calls_from_response(
        ChatResponse(message=message, raw=_raw(usage_only=True))
    )
    assert len(calls) == 1


def test_complete_tool_call_unchanged(llm: OpenAI) -> None:
    calls = llm.get_tool_calls_from_response(_response("tool_calls"))
    assert len(calls) == 1
    assert calls[0].tool_name == "write_file"
