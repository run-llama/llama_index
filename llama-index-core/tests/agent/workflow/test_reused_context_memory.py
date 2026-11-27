"""
Tests for the `memory` passed to `run()` when a `Context` is reused.

Reusing a `Context` continues from a serialized snapshot of the previous run, so the
memory held by the context is a deserialized copy, not the object the caller passed.
The memory passed to `run()` must still be the one that run reads and updates.
"""

from typing import Any, Callable, List, Tuple

import httpx
import pytest

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent, ReActAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer, Memory
from llama_index.core.workflow import Context

_PROVIDER_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "COHERE_API_KEY",
    "MISTRAL_API_KEY",
)


@pytest.fixture(autouse=True)
def no_real_api_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make real LLM/API calls impossible, whatever the environment contains."""
    for name in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    def deny_http(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network access is not allowed in these tests")

    monkeypatch.setattr(httpx.Client, "send", deny_http)
    monkeypatch.setattr(httpx.AsyncClient, "send", deny_http)


def _mock_llm(text: str = "ok") -> MockFunctionCallingLLM:
    return MockFunctionCallingLLM(
        response_generator=lambda messages, **kwargs: ChatMessage(
            role=MessageRole.ASSISTANT, content=text
        )
    )


def _function_agent() -> FunctionAgent:
    return FunctionAgent(
        name="agent",
        description="answers",
        llm=_mock_llm(),
        tools=[],
        streaming=False,
    )


def _react_agent() -> ReActAgent:
    return ReActAgent(
        name="agent",
        description="answers",
        llm=_mock_llm("Thought: I can answer directly.\nAnswer: ok"),
        tools=[],
        streaming=False,
    )


def _user_turns(memory: BaseMemory) -> List[str]:
    return [
        str(message.content)
        for message in memory.get_all()
        if message.role == MessageRole.USER
    ]


def _recording_agent() -> Tuple[FunctionAgent, List[List[ChatMessage]]]:
    """A FunctionAgent that records the messages its LLM receives on every call."""
    seen: List[List[ChatMessage]] = []

    def generate(messages: List[ChatMessage], **kwargs: Any) -> ChatMessage:
        seen.append(list(messages))
        return ChatMessage(role=MessageRole.ASSISTANT, content="ok")

    agent = FunctionAgent(
        name="agent",
        description="answers",
        llm=MockFunctionCallingLLM(response_generator=generate),
        tools=[],
        streaming=False,
    )
    return agent, seen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_agent", [_function_agent, _react_agent], ids=["function", "react"]
)
async def test_reused_context_writes_to_supplied_memory(
    make_agent: Callable[[], Any],
) -> None:
    agent = make_agent()
    memory = ChatMemoryBuffer.from_defaults()
    ctx = Context(agent)

    await agent.run(user_msg="first", memory=memory, ctx=ctx)
    await agent.run(user_msg="second", memory=memory, ctx=ctx)

    assert await ctx.store.get("memory") is memory
    assert _user_turns(memory) == ["first", "second"]
    assert len(memory.get_all()) == 4


@pytest.mark.asyncio
async def test_agent_workflow_reused_context_writes_to_supplied_memory() -> None:
    workflow = AgentWorkflow(agents=[_function_agent()], root_agent="agent")
    memory = ChatMemoryBuffer.from_defaults()
    ctx = Context(workflow)

    await workflow.run(user_msg="first", memory=memory, ctx=ctx)
    await workflow.run(user_msg="second", memory=memory, ctx=ctx)

    assert await ctx.store.get("memory") is memory
    assert _user_turns(memory) == ["first", "second"]
    assert len(memory.get_all()) == 4


@pytest.mark.asyncio
async def test_restored_context_writes_to_supplied_memory() -> None:
    agent = _function_agent()
    memory = ChatMemoryBuffer.from_defaults()
    ctx = Context(agent)
    await agent.run(user_msg="first", memory=memory, ctx=ctx)

    restored_ctx = Context.from_dict(agent, ctx.to_dict())
    await agent.run(user_msg="second", memory=memory, ctx=restored_ctx)

    assert await restored_ctx.store.get("memory") is memory
    assert _user_turns(memory) == ["first", "second"]


@pytest.mark.asyncio
async def test_new_memory_replaces_memory_held_by_reused_context() -> None:
    """
    An explicit memory passed on a later run replaces the context's memory.

    The context's earlier history is not carried over: the new memory is the
    source of truth for that run.
    """
    agent, seen = _recording_agent()
    ctx = Context(agent)
    old_memory = ChatMemoryBuffer.from_defaults()
    await agent.run(user_msg="first", memory=old_memory, ctx=ctx)

    new_memory = ChatMemoryBuffer.from_defaults()
    await agent.run(user_msg="second", memory=new_memory, ctx=ctx)

    assert await ctx.store.get("memory") is new_memory
    assert _user_turns(new_memory) == ["second"]
    assert len(new_memory.get_all()) == 2
    # the replaced memory is no longer written to
    assert _user_turns(old_memory) == ["first"]
    # the LLM saw the new memory's content only, not the earlier turn
    last_call = [str(message.content) for message in seen[-1]]
    assert "second" in last_call
    assert "first" not in last_call


@pytest.mark.asyncio
async def test_new_memory_with_history_is_used_on_reused_context() -> None:
    agent, seen = _recording_agent()
    ctx = Context(agent)
    await agent.run(user_msg="first", ctx=ctx)

    new_memory = ChatMemoryBuffer.from_defaults()
    await new_memory.aput(ChatMessage(role=MessageRole.USER, content="earlier"))
    await new_memory.aput(ChatMessage(role=MessageRole.ASSISTANT, content="noted"))
    await agent.run(user_msg="second", memory=new_memory, ctx=ctx)

    assert await ctx.store.get("memory") is new_memory
    assert _user_turns(new_memory) == ["earlier", "second"]
    last_call = [str(message.content) for message in seen[-1]]
    assert "earlier" in last_call
    assert "first" not in last_call


@pytest.mark.asyncio
async def test_first_run_stores_supplied_memory_by_identity() -> None:
    agent = _function_agent()
    memory = ChatMemoryBuffer.from_defaults()
    ctx = Context(agent)

    await agent.run(user_msg="first", memory=memory, ctx=ctx)

    assert await ctx.store.get("memory") is memory
    assert _user_turns(memory) == ["first"]


@pytest.mark.asyncio
async def test_reused_context_without_memory_keeps_history() -> None:
    agent = _function_agent()
    ctx = Context(agent)

    await agent.run(user_msg="first", ctx=ctx)
    await agent.run(user_msg="second", ctx=ctx)

    stored_memory = await ctx.store.get("memory")
    assert _user_turns(stored_memory) == ["first", "second"]


@pytest.mark.asyncio
async def test_omitting_memory_on_later_run_keeps_context_history() -> None:
    agent = _function_agent()
    memory = ChatMemoryBuffer.from_defaults()
    ctx = Context(agent)

    await agent.run(user_msg="first", memory=memory, ctx=ctx)
    await agent.run(user_msg="second", ctx=ctx)

    stored_memory = await ctx.store.get("memory")
    assert _user_turns(stored_memory) == ["first", "second"]


@pytest.mark.asyncio
async def test_memory_class_works_on_reused_context() -> None:
    agent = _function_agent()
    memory = Memory.from_defaults(session_id="reused-context-memory")
    ctx = Context(agent)

    await agent.run(user_msg="first", memory=memory, ctx=ctx)
    await agent.run(user_msg="second", memory=memory, ctx=ctx)

    messages = await memory.aget_all()
    assert [m.content for m in messages if m.role == MessageRole.USER] == [
        "first",
        "second",
    ]
