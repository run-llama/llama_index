import json
from typing import Any

import pytest

from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from llama_index.tools.mcp.utils import workflow_as_mcp


class TenantStart(StartEvent):
    tenant_id: str


class CountingWorkflow(Workflow):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.call_count = 0
        self.history: list[str] = []

    @step
    async def echo(self, ctx: Context, ev: TenantStart) -> StopEvent:
        self.call_count += 1
        self.history.append(ev.tenant_id)
        return StopEvent(
            result={
                "call_index": self.call_count,
                "history_visible": list(self.history),
            }
        )


@pytest.mark.asyncio
async def test_workflow_as_mcp_factory_creates_workflow_per_call() -> None:
    created_workflows: list[CountingWorkflow] = []

    def workflow_factory() -> CountingWorkflow:
        workflow = CountingWorkflow(timeout=5)
        created_workflows.append(workflow)
        return workflow

    app = workflow_as_mcp(
        workflow_factory=workflow_factory,
        workflow_name="CountingWorkflow",
        start_event_model=TenantStart,
    )

    alice_result = await app.call_tool(
        "CountingWorkflow", {"run_args": {"tenant_id": "alice"}}
    )
    bob_result = await app.call_tool(
        "CountingWorkflow", {"run_args": {"tenant_id": "bob"}}
    )

    alice_payload = json.loads(alice_result[0].text)
    bob_payload = json.loads(bob_result[0].text)

    assert alice_payload == {"call_index": 1, "history_visible": ["alice"]}
    assert bob_payload == {"call_index": 1, "history_visible": ["bob"]}
    assert [workflow.history for workflow in created_workflows] == [["alice"], ["bob"]]
