from typing import Any

import pytest

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from llama_index.tools.mcp import workflow_as_mcp


class TenantStartEvent(StartEvent):
    tenant_id: str


class CountingWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__(timeout=30)
        self.history: list[str] = []

    @step
    async def count(self, event: TenantStartEvent) -> StopEvent:
        self.history.append(event.tenant_id)
        return StopEvent(result=list(self.history))


async def _call_workflow(app: Any, tenant_id: str) -> list[str]:
    result = await app.call_tool(
        "CountingWorkflow",
        {"run_args": {"tenant_id": tenant_id}},
    )
    return [item.text for item in result]


@pytest.mark.asyncio
async def test_workflow_factory_isolates_invocations() -> None:
    template = CountingWorkflow()
    instances: list[CountingWorkflow] = []

    def workflow_factory() -> CountingWorkflow:
        workflow = CountingWorkflow()
        instances.append(workflow)
        return workflow

    app = workflow_as_mcp(template, workflow_factory=workflow_factory)

    assert await _call_workflow(app, "alice") == ["alice"]
    assert await _call_workflow(app, "bob") == ["bob"]
    assert len(instances) == 2
    assert template.history == []


@pytest.mark.asyncio
async def test_workflow_instance_retains_existing_reuse_behavior() -> None:
    workflow = CountingWorkflow()
    app = workflow_as_mcp(workflow)

    assert await _call_workflow(app, "alice") == ["alice"]
    assert await _call_workflow(app, "bob") == ["alice", "bob"]
    assert workflow.history == ["alice", "bob"]
