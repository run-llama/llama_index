import pytest
from typing import Sequence, Any
from pydantic import BaseModel, Field

from llama_index.core.workflow import Context
from llama_index.core.agent.workflow.base_agent import _safe_model_dump
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatInProgressEvent,
    LLMCompletionEndEvent,
    LLMCompletionInProgressEvent,
)
from llama_index.core.memory import ChatMemoryBuffer


class DashScopeLikeResponse(dict):
    """
    Simulates dashscope.api_entities.dashscope_response.DashScopeResponse
    where __getattr__ raises KeyError instead of AttributeError for missing attributes.
    """

    def __getattr__(self, attr: str) -> Any:
        return self[attr]


class SampleModel(BaseModel):
    foo: str = Field(default="bar")


def test_safe_model_dump():
    # 1. DashScope-like response with faulty __getattr__
    dashscope_raw = DashScopeLikeResponse({"status_code": 200, "output": {"text": "hello"}})
    res = _safe_model_dump(dashscope_raw)
    assert res == dashscope_raw
    assert isinstance(res, DashScopeLikeResponse)

    # 2. Pydantic BaseModel
    model_raw = SampleModel(foo="test")
    res = _safe_model_dump(model_raw)
    assert res == {"foo": "test"}

    # 3. Primitives and None
    assert _safe_model_dump(None) is None
    assert _safe_model_dump({"a": 1}) == {"a": 1}
    assert _safe_model_dump("text") == "text"
    assert _safe_model_dump(123) == 123


def test_instrumentation_events_with_dashscope_like_raw():
    dashscope_raw = DashScopeLikeResponse({"status_code": 200, "output": {"text": "hello"}})

    # LLMCompletionInProgressEvent
    event1 = LLMCompletionInProgressEvent(
        prompt="test",
        response=CompletionResponse(text="output", raw=dashscope_raw),
    )
    dumped1 = event1.model_dump()
    assert dumped1["response"]["raw"] == dashscope_raw

    # LLMCompletionEndEvent
    event2 = LLMCompletionEndEvent(
        prompt="test",
        response=CompletionResponse(text="output", raw=dashscope_raw),
    )
    dumped2 = event2.model_dump()
    assert dumped2["response"]["raw"] == dashscope_raw

    # LLMChatInProgressEvent
    event3 = LLMChatInProgressEvent(
        messages=[ChatMessage(role=MessageRole.USER, content="hi")],
        response=ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content="hi"), raw=dashscope_raw),
    )
    dumped3 = event3.model_dump()
    assert dumped3["response"]["raw"] == dashscope_raw

    # LLMChatEndEvent
    event4 = LLMChatEndEvent(
        messages=[ChatMessage(role=MessageRole.USER, content="hi")],
        response=ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content="hi"), raw=dashscope_raw),
    )
    dumped4 = event4.model_dump()
    assert dumped4["response"]["raw"] == dashscope_raw


from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import LLMMetadata


class DashScopeMockLLM(LLM):
    """Mock LLM returning DashScopeLikeResponse in raw."""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=False)

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raw = DashScopeLikeResponse({"status_code": 200, "output": {"text": "Thought: none\nAnswer: 42"}})
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="Thought: none\nAnswer: 42"),
            raw=raw,
        )

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        raw = DashScopeLikeResponse({"status_code": 200, "output": {"text": "Thought: none\nAnswer: 42"}})

        async def gen():
            yield ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content="Thought: none\nAnswer: 42"),
                delta="Thought: none\nAnswer: 42",
                raw=raw,
            )

        return gen()

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content="Thought: none\nAnswer: 42"))

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text="Thought: none\nAnswer: 42")

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        raise NotImplementedError

    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text="Thought: none\nAnswer: 42")

    async def astream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_react_agent_with_dashscope_like_raw():
    llm = DashScopeMockLLM()
    agent = ReActAgent(llm=llm, tools=[])
    ctx = Context(agent)
    memory = ChatMemoryBuffer.from_defaults()

    output = await agent.take_step(
        ctx=ctx,
        llm_input=[ChatMessage(role=MessageRole.USER, content="What is 20 + 22?")],
        tools=[],
        memory=memory,
    )

    assert output.response.content == "Thought: none\nAnswer: 42"
    assert isinstance(output.raw, DashScopeLikeResponse)
    assert output.raw["status_code"] == 200


@pytest.mark.asyncio
async def test_react_agent_streaming_with_dashscope_like_raw():
    llm = DashScopeMockLLM()
    agent = ReActAgent(llm=llm, tools=[], streaming=True)
    ctx = Context(agent)
    memory = ChatMemoryBuffer.from_defaults()

    output = await agent.take_step(
        ctx=ctx,
        llm_input=[ChatMessage(role=MessageRole.USER, content="What is 20 + 22?")],
        tools=[],
        memory=memory,
    )

    assert output.response.content == "Thought: none\nAnswer: 42"
    assert isinstance(output.raw, DashScopeLikeResponse)
    assert output.raw["status_code"] == 200

