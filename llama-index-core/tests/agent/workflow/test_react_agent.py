from typing import Any, AsyncGenerator, Sequence

import pytest

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.agent.workflow.workflow_events import AgentStream
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.bridge.pydantic import BaseModel, PrivateAttr
from llama_index.core.llms import MockLLM
from llama_index.core.prompts import PromptTemplate


class RawResponseLLM(MockLLM):
    _raw: Any = PrivateAttr()

    def __init__(self, raw: Any) -> None:
        super().__init__(is_chat_model=True)
        self._raw = raw

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        content = "Thought: I can answer\nAnswer: done"
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            raw=self._raw,
        )

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> AsyncGenerator[ChatResponse, None]:
        content = "Thought: I can answer\nAnswer: done"

        async def gen() -> AsyncGenerator[ChatResponse, None]:
            yield ChatResponse(
                message=ChatMessage(role="assistant", content=content),
                delta=content,
                raw=self._raw,
            )

        return gen()


class PydanticRaw(BaseModel):
    value: str


class FailingDumpRaw(BaseModel):
    value: str

    def model_dump(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("dump failed")


class ProblematicRaw:
    def __getattr__(self, name: str) -> Any:
        if name == "__pydantic_validator__":
            raise KeyError(name)

        raise AttributeError(name)


def test_react_agent_prompts():
    llm = MockLLM()
    agent = ReActAgent(
        llm=llm,
        tools=[],
    )

    prompts = agent.get_prompts()
    assert len(prompts) == 1
    assert isinstance(prompts["react_header"], PromptTemplate)

    new_prompt = "New prompt"
    agent.update_prompts({"react_header": new_prompt})
    prompts = agent.get_prompts()
    assert len(prompts) == 1
    assert new_prompt in str(prompts["react_header"])

    new_prompt = PromptTemplate("New prompt 2")
    agent.update_prompts({"react_header": new_prompt})
    prompts = agent.get_prompts()
    assert len(prompts) == 1
    assert new_prompt == prompts["react_header"]


@pytest.mark.asyncio
async def test_react_agent_dumps_pydantic_raw_response():
    raw = PydanticRaw(value="ok")
    agent = ReActAgent(llm=RawResponseLLM(raw), tools=[])

    response = await agent.run(user_msg="hello")

    assert response.raw == {"value": "ok"}


@pytest.mark.asyncio
async def test_react_agent_preserves_raw_response_when_model_dump_fails():
    raw = FailingDumpRaw(value="ok")
    agent = ReActAgent(llm=RawResponseLLM(raw), tools=[])

    response = await agent.run(user_msg="hello")

    assert response.raw is raw


@pytest.mark.asyncio
async def test_react_agent_preserves_problematic_raw_response_object():
    raw = ProblematicRaw()
    agent = ReActAgent(llm=RawResponseLLM(raw), tools=[], streaming=True)

    handler = agent.run(user_msg="hello")
    streamed_raw = []
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            streamed_raw.append(event.raw)

    response = await handler

    assert streamed_raw == [raw]
    assert response.raw is raw
