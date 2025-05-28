import random
from typing import Annotated, List, Optional

import pytest
from llama_index.core.llms import ChatMessage, MockLLM
from llama_index.core.memory import Memory
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.core.workflow.types import Resource
from llama_index.core.workflow.workflow import Workflow
from pydantic import BaseModel, Field


class SecondEvent(Event):
    msg: str = Field(description="A message")


class ThirdEvent(Event):
    msg: str = Field(description="A message")


class MessageHistory(BaseModel):
    messages: List[ChatMessage] = Field(
        default_factory=list,
        description="Messages",
    )

    def put(self, message: ChatMessage):
        self.messages.append(message)

    def get(self):
        return self.messages


class MessageStopEvent(StopEvent):
    llm_response: Optional[str] = Field(default=None)


@pytest.mark.asyncio
async def test_resource():
    m = Memory.from_defaults("user_id_123", token_limit=60000)

    def get_memory(*args, **kwargs) -> Memory:
        return m

    class TestWorkflow(Workflow):
        @step
        def start_step(self, ev: StartEvent) -> SecondEvent:
            print("Start step is done", flush=True)
            return SecondEvent(msg="Hello")

        @step
        def second_step(self, ev: SecondEvent) -> ThirdEvent | StopEvent:
            if ev.msg == "Hello":
                return ThirdEvent(msg="Hello 2")
            return StopEvent()

        @step
        def f1(
            self, ev: ThirdEvent, memory: Annotated[Memory, Resource(get_memory)]
        ) -> SecondEvent:
            memory.put(ChatMessage.from_str(role="user", content=ev.msg))
            if random.randint(0, 1) == 0:
                return SecondEvent(msg="Hello")
            else:
                return SecondEvent(msg="Hello 3")

    wf = TestWorkflow(disable_validation=True)
    await wf.run()
    mem = m.get()
    assert len(mem) >= 1
    assert all(el.blocks[0].text == "Hello 2" for el in mem)


@pytest.mark.asyncio
async def test_resource_async():
    async def hello_world():
        return "Hello world!"

    async def create_message_history(*args, **kwargs) -> MessageHistory:
        first_message = await hello_world()
        return MessageHistory(
            messages=[ChatMessage.from_str(content=first_message, role="user")]
        )

    class TestWorkflow(Workflow):
        @step
        def start_step(self, ev: StartEvent) -> SecondEvent:
            print("Start step is done", flush=True)
            return SecondEvent(msg="Hello")

        @step
        def second_step(self, ev: SecondEvent) -> ThirdEvent | StopEvent:
            if ev.msg == "Hello":
                return ThirdEvent(msg="Hello world!")
            return StopEvent()

        @step
        def f1(
            self,
            ev: ThirdEvent,
            history: Annotated[MessageHistory, create_message_history],
        ) -> SecondEvent:
            global m
            history.put(ChatMessage.from_str(role="user", content=ev.msg))
            m = history
            if random.randint(0, 1) == 0:
                return SecondEvent(msg="Hello")
            else:
                return SecondEvent(msg="Hello back!")

    wf = TestWorkflow(disable_validation=True)
    await wf.run()
    mem = m.get()
    assert len(mem) >= 2
    assert all(el.blocks[0].text == "Hello world!" for el in mem)


@pytest.mark.asyncio
async def test_resource_with_llm():
    def get_llm(*args, **kwargs) -> MockLLM:
        return MockLLM()

    class TestWorkflow(Workflow):
        @step
        def test_step(
            self, ev: StartEvent, llm: Annotated[MockLLM, get_llm]
        ) -> MessageStopEvent:
            response = llm.complete("Hey there, who are you?")
            res = response.text or None
            return MessageStopEvent(llm_response=res)

    wf = TestWorkflow(disable_validation=True)
    handler = await wf.run()
    response = handler.llm_response
    assert response is not None
    assert isinstance(response, str)
