import random
from typing import Annotated, List, Optional, Union

import pytest
from llama_index.core.llms import ChatMessage, MockLLM
from llama_index.core.memory import Memory
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.core.workflow.resource import Resource
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
        def second_step(self, ev: SecondEvent) -> Union[ThirdEvent,StopEvent]:
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
        def second_step(self, ev: SecondEvent) -> Union[ThirdEvent, StopEvent]:
            if ev.msg == "Hello":
                return ThirdEvent(msg="Hello world!")
            return StopEvent()

        @step
        def f1(
            self,
            ev: ThirdEvent,
            history: Annotated[MessageHistory, Resource(create_message_history)],
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
            self,
            ev: StartEvent,
            llm: Annotated[MockLLM, Resource(get_llm, cache=False)],
        ) -> MessageStopEvent:
            response = llm.complete("Hey there, who are you?")
            res = response.text or None
            return MessageStopEvent(llm_response=res)

    wf = TestWorkflow(disable_validation=True)
    handler = await wf.run()
    response = handler.llm_response
    assert response is not None
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_caching_behavior():
    class CounterThing:
        counter = 0

        def incr(self):
            self.counter += 1

    class StepEvent(Event):
        pass

    def provide_counter_thing() -> CounterThing:
        return CounterThing()

    class TestWorkflow(Workflow):
        @step
        async def test_step(
            self,
            ev: StartEvent,
            counter_thing: Annotated[CounterThing, Resource(provide_counter_thing)],
        ) -> StepEvent:
            counter_thing.incr()
            return StepEvent()

        @step
        async def test_step_2(
            self,
            ev: StepEvent,
            counter_thing: Annotated[CounterThing, Resource(provide_counter_thing)],
        ) -> StopEvent:
            global cc
            counter_thing.incr()
            cc = counter_thing.counter
            return StopEvent()

    wf_1 = TestWorkflow(disable_validation=True)
    await wf_1.run()
    assert (
        cc == 2
    )  # this is expected to be 2, as it is a cached resource shared by test_step and test_step_2, which means at test_step it counter_thing.counter goes from 0 to 1 and at test_step_2 goes from 1 to 2

    wf_2 = TestWorkflow(disable_validation=True)
    await wf_2.run()
    assert (
        cc == 2
    )  # the cache is workflow-specific, so since wf_2 is different from wf_1, we expect no interference between the two


@pytest.mark.asyncio
async def test_non_caching_behavior():
    class CounterThing:
        counter = 0

        def incr(self):
            self.counter += 1

    class StepEvent(Event):
        pass

    def provide_counter_thing() -> CounterThing:
        return CounterThing()

    class TestWorkflow(Workflow):
        @step
        async def test_step(
            self,
            ev: StartEvent,
            counter_thing: Annotated[CounterThing, Resource(provide_counter_thing)],
        ) -> StepEvent:
            global cc1
            counter_thing.incr()
            cc1 = counter_thing.counter
            return StepEvent()

        @step
        async def test_step_2(
            self,
            ev: StepEvent,
            counter_thing: Annotated[
                CounterThing, Resource(provide_counter_thing, cache=False)
            ],
        ) -> StopEvent:
            global cc2
            counter_thing.incr()
            cc2 = counter_thing.counter
            return StopEvent()

    wf_1 = TestWorkflow(disable_validation=True)
    await wf_1.run()
    assert cc1 == 1
    assert cc2 == 1
