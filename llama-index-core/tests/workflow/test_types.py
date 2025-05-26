import queue

import pytest
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import StartEvent, StopEvent
from llama_index.core.workflow.types import Resource
from llama_index.core.workflow.workflow import Workflow


@pytest.mark.asyncio
async def test_resource():
    q = queue.Queue()

    def get_memory(*args, **kwargs) -> queue.Queue:
        return q

    class TestWorkflow(Workflow):
        @step
        def f1(
            self, ev: StartEvent, memory: Resource[queue.Queue, get_memory]
        ) -> StopEvent:
            memory.put("test data")
            return StopEvent()

    wf = TestWorkflow(disable_validation=True)
    await wf.run()
    assert q.get() == "test data"
