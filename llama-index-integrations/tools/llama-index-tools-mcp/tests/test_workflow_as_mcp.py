import json
from typing import Any

import pytest
from pydantic import BaseModel

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

    first_payload = _tool_payload(first_result)
    second_payload = _tool_payload(second_result)

    # Each call runs on a fresh instance, so nothing accumulates on `self`.
    assert first_payload["call_index"] == 1
    assert second_payload["call_index"] == 1
    assert first_payload["history_visible"] == ["alice"]
    assert second_payload["history_visible"] == ["bob"]

    assert first_payload["label"] == "factory"
    assert second_payload["label"] == "factory"
    assert factory_call_count == 2


@pytest.mark.asyncio
async def test_workflow_as_mcp_shares_instance_without_factory() -> None:
    app = workflow_as_mcp(CountingWorkflow(label="shared", timeout=30))

    first_result = await app.call_tool(
        "CountingWorkflow", {"run_args": {"tenant_id": "alice"}}
    )
    second_result = await app.call_tool(
        "CountingWorkflow", {"run_args": {"tenant_id": "bob"}}
    )

    # Characterization test: documents the pre-existing shared-instance behavior
    # that `workflow_factory` opts out of. This is not a guarantee; drop this
    # test if the default ever becomes per-call isolation.
    assert _tool_payload(first_result)["call_index"] == 1
    assert _tool_payload(second_result)["call_index"] == 2
    assert _tool_payload(second_result)["history_visible"] == ["alice", "bob"]


@pytest.mark.asyncio
async def test_workflow_as_mcp_accepts_async_workflow_factory() -> None:
    factory_call_count = 0

    async def workflow_factory() -> CountingWorkflow:
        nonlocal factory_call_count
        factory_call_count += 1
        return CountingWorkflow(label="async-factory", timeout=30)

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

    first_payload = _tool_payload(first_result)
    second_payload = _tool_payload(second_result)

    # Awaited factories get the same per-call isolation as sync ones.
    assert first_payload["call_index"] == 1
    assert second_payload["call_index"] == 1
    assert first_payload["history_visible"] == ["alice"]
    assert second_payload["history_visible"] == ["bob"]

    assert first_payload["label"] == "async-factory"
    assert second_payload["label"] == "async-factory"
    assert factory_call_count == 2


@pytest.mark.asyncio
async def test_workflow_as_mcp_uses_factory_for_base_model_start_event() -> None:
    """The factory must also be honored on the plain-BaseModel dispatch branch."""

    class TenantArgs(BaseModel):
        tenant_id: str

    factory_call_count = 0

    def workflow_factory() -> CountingWorkflow:
        nonlocal factory_call_count
        factory_call_count += 1
        return CountingWorkflow(label="factory", timeout=30)

    app = workflow_as_mcp(
        CountingWorkflow(label="metadata", timeout=30),
        start_event_model=TenantArgs,
        workflow_factory=workflow_factory,
    )

    first_result = await app.call_tool(
        "CountingWorkflow", {"run_args": {"tenant_id": "alice"}}
    )
    second_result = await app.call_tool(
        "CountingWorkflow", {"run_args": {"tenant_id": "bob"}}
    )

    first_payload = _tool_payload(first_result)
    second_payload = _tool_payload(second_result)

    assert first_payload["call_index"] == 1
    assert second_payload["call_index"] == 1
    assert second_payload["history_visible"] == ["bob"]
    assert second_payload["label"] == "factory"
    assert factory_call_count == 2
