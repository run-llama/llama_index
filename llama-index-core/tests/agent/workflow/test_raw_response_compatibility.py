"""Raw response compatibility with providers exposing dictionary-style attributes."""

from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, PrivateAttr

from llama_index.core.agent.workflow import (
    AgentStream,
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)
from llama_index.core.agent.workflow.codeact_agent import CodeActAgent
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.workflow import Context


class ProviderRaw(dict):
    def __getattr__(self, name):
        return self[name]


class ModelRaw(BaseModel):
    value: str = "answer"


class UnserializableRaw(BaseModel):
    def model_dump(self, *args, **kwargs):
        raise ValueError("raw serialization failed")


class RawResponseLLM(MockFunctionCallingLLM):
    _raw: Any = PrivateAttr()
    _contents: list[str] = PrivateAttr()

    def __init__(self, raw, contents=None):
        super().__init__(is_chat_model=True)
        self._raw = raw
        self._contents = list(contents or ["Thought: I can answer.\nAnswer: done"])

    def _response(self):
        content = (
            self._contents.pop(0) if len(self._contents) > 1 else self._contents[0]
        )
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            delta=content,
            raw=self._raw,
        )

    async def achat(self, messages, **kwargs):
        return self._response()

    async def astream_chat(self, messages, **kwargs):
        async def responses():
            yield self._response()

        return responses()


def make_raw(kind):
    if kind == "provider":
        return ProviderRaw(value="answer")
    if kind == "model":
        return ModelRaw()
    if kind == "dict":
        return {"value": "answer"}
    if kind == "unserializable":
        return UnserializableRaw()
    return None


def make_agent(kind, raw, streaming, contents=None):
    llm = RawResponseLLM(raw, contents)
    if kind == "react":
        return ReActAgent(llm=llm, tools=[], streaming=streaming, timeout=5)
    if kind == "function":
        return FunctionAgent(llm=llm, tools=[], streaming=streaming, timeout=5)

    def execute(code: str) -> str:
        raise AssertionError("This response must not execute code")

    return CodeActAgent(code_execute_fn=execute, llm=llm, streaming=streaming)


def assert_raw(actual, raw):
    if type(raw) is ModelRaw:
        assert actual == {"value": "answer"}
        assert type(actual) is dict
    else:
        assert actual is raw


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_kind", ["react", "function", "codeact"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("raw_kind", ["provider", "model", "dict", "none"])
async def test_public_agent_raw_response(agent_kind, streaming, raw_kind):
    raw = make_raw(raw_kind)
    agent = make_agent(agent_kind, raw, streaming)
    handler = agent.run(user_msg="hello")
    events = [
        event
        async for event in handler.stream_events()
        if isinstance(event, AgentStream)
    ]
    output = await handler

    assert "done" in str(output)
    assert_raw(output.raw, raw)
    if streaming:
        assert events
        for event in events:
            assert_raw(event.raw, raw)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_kind", ["react", "function", "codeact"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_public_agent_preserves_serialization_error(agent_kind, streaming):
    agent = make_agent(agent_kind, UnserializableRaw(), streaming)

    with pytest.raises(ValueError, match="raw serialization failed"):
        await agent.run(user_msg="hello")


@pytest.mark.asyncio
@pytest.mark.parametrize("first_content", ["", "Thought: still thinking"])
async def test_react_retry_preserves_provider_raw(first_content):
    raw = ProviderRaw(value="answer")
    agent = make_agent(
        "react", raw, False, [first_content, "Thought: I can answer.\nAnswer: done"]
    )
    output = await agent.run(user_msg="hello")

    assert str(output) == "done"
    assert output.raw is raw


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_kind", ["single", "multi"])
@pytest.mark.parametrize(
    "raw_kind", ["provider", "model", "dict", "none", "unserializable"]
)
async def test_workflow_response_stream_raw(workflow_kind, raw_kind):
    raw = make_raw(raw_kind)
    agent = make_agent("react", raw, True)
    # Only the event sink is mocked; both workflow objects and response methods are real.
    ctx = Mock(spec=Context)
    ctx.is_running = True
    if workflow_kind == "single":
        response_call = agent._get_llm_response(
            ctx, [ChatMessage(role="user", content="hello")]
        )
    else:
        workflow = AgentWorkflow(agents=[agent])
        response_call = workflow._get_llm_response(
            ctx, [ChatMessage(role="user", content="hello")], agent
        )

    if raw_kind == "unserializable":
        with pytest.raises(ValueError, match="raw serialization failed"):
            await response_call
    else:
        response = await response_call
        assert response.raw is raw
        ctx.write_event_to_stream.assert_called_once()
        assert_raw(ctx.write_event_to_stream.call_args.args[0].raw, raw)
