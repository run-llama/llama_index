from __future__ import annotations

import pytest
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import (
    StartEvent,
    StopEvent,
)
from llama_index.core.workflow.workflow import Workflow

from .conftest import OneTestEvent


class PostponedAnnotationsWorkflow(Workflow):
    @step
    async def step1(self, ev: StartEvent) -> OneTestEvent:
        return OneTestEvent(test_param="postponed")

    @step
    async def step2(self, ev: OneTestEvent) -> StopEvent:
        return StopEvent(result=f"Handled {ev.test_param}")


@pytest.mark.asyncio()
async def test_workflow_postponed_annotations():
    workflow = PostponedAnnotationsWorkflow()
    result = await workflow.run()
    assert workflow.is_done()
    assert result == "Handled postponed"


@pytest.mark.asyncio()
async def test_workflow_forward_reference():
    class ForwardRefWorkflow(Workflow):
        @step
        async def step1(self, ev: StartEvent) -> OneTestEvent:
            return OneTestEvent(test_param="forward")

        @step
        async def step2(self, ev: OneTestEvent) -> StopEvent:
            return StopEvent(result=f"Handled {ev.test_param}")

    workflow = ForwardRefWorkflow()
    result = await workflow.run()
    assert workflow.is_done()
    assert result == "Handled forward"
