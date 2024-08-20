import pytest
from typing import Union, Optional

from llama_index.core.workflow.workflow import (
    Workflow,
    Context,
)
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import StartEvent, StopEvent

from .conftest import OneTestEvent, AnotherTestEvent


@pytest.mark.asyncio()
async def test_collect_events():
    ev1 = OneTestEvent()
    ev2 = AnotherTestEvent()

    class TestWorkflow(Workflow):
        @step()
        async def step1(self, _: StartEvent) -> OneTestEvent:
            return ev1

        @step()
        async def step2(self, _: StartEvent) -> AnotherTestEvent:
            return ev2

        @step(pass_context=True)
        async def step3(
            self, ctx: Context, ev: Union[OneTestEvent, AnotherTestEvent]
        ) -> Optional[StopEvent]:
            events = ctx.collect_events(ev, [OneTestEvent, AnotherTestEvent])
            if events is None:
                return None
            return StopEvent(result=events)

    workflow = TestWorkflow()
    result = await workflow.run()
    assert result == [ev1, ev2]


@pytest.mark.asyncio()
async def test_set_global():
    c1 = Context()
    await c1.set(key="test_key", value=42)

    c2 = Context(parent=c1)
    assert await c2.get(key="test_key") == 42


@pytest.mark.asyncio()
async def test_set_private():
    c1 = Context()
    await c1.set(key="test_key", value=42, make_private=True)
    assert await c1.get(key="test_key") == 42

    c2 = Context(parent=c1)
    with pytest.raises(ValueError):
        await c2.get(key="test_key")


@pytest.mark.asyncio()
async def test_set_private_duplicate():
    c1 = Context()
    await c1.set(key="test_key", value=42)

    c2 = Context(parent=c1)
    with pytest.raises(ValueError):
        await c2.set(key="test_key", value=99, make_private=True)


@pytest.mark.asyncio()
async def test_get_default():
    c1 = Context()
    assert await c1.get(key="test_key", default=42) == 42


@pytest.mark.asyncio()
async def test_legacy_data():
    c1 = Context()
    await c1.set(key="test_key", value=42)
    assert c1.data["test_key"] == 42
