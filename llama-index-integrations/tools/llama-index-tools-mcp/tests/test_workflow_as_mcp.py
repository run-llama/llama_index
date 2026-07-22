import json
from typing import Any

import pytest

from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from llama_index.tools.mcp import workflow_as_mcp


class TenantStart(StartEvent):
    tenant_id: str


class CountingWorkflow(Workflow):
    def __init__(self, label: str = "default", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.label = label
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
                "label": self.label,
            }
        )


def _tool_payload(result: list[Any]) -> dict[str, Any]:
    return json.loads(result[0].text)


@pytest.mark.asyncio
async def test_workflow_as_mcp_accepts_explicit_workflow_factory() -> None:
    factory_call_count = 0

    def workflow_factory() -> CountingWorkflow:
        nonlocal factory_call_count
        factory_call_count += 1
        return CountingWorkflow(label="factory", timeout=30)

    app = workflow_as_mcp(
        CountingWorkflow(label="metadata", timeout=30),
        workflow_factory=workflow_factory,
    )

    first_result = await app.call_tool(
        "CountingWorkflow", {"run_args": {"tenant_id": "alice"}}
    )
    second_result = await app.call_tool(
        "CountingWorkflow", {"run_args": {"tenant_id": "bob"}}
    )

    assert _tool_payload(first_result)["label"] == "factory"
    assert _tool_payload(second_result)["label"] == "factory"
    assert factory_call_count == 2
