import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from llama_index.protocols.ag_ui.events import (
    ActivityDeltaWorkflowEvent,
    ActivitySnapshotWorkflowEvent,
)
from llama_index.protocols.ag_ui.router import AGUIWorkflowRouter

OPERATIONS = [
    {"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": "c1"}},
]


class ActivityWorkflow(Workflow):
    @step
    async def emit(self, ctx: Context, ev: StartEvent) -> StopEvent:
        ctx.write_event_to_stream(
            ActivitySnapshotWorkflowEvent(
                message_id="a1",
                activity_type="a2ui-surface",
                content={"a2ui_operations": OPERATIONS},
            )
        )
        ctx.write_event_to_stream(
            ActivityDeltaWorkflowEvent(
                message_id="a1",
                activity_type="a2ui-surface",
                patch=[{"op": "add", "path": "/status", "value": "done"}],
            )
        )
        return StopEvent()


async def _factory() -> Workflow:
    return ActivityWorkflow()


def _sse_events(body: str) -> list:
    return [
        json.loads(line[len("data:") :])
        for line in body.splitlines()
        if line.startswith("data:")
    ]


def test_router_streams_activity_events():
    app = FastAPI()
    app.include_router(AGUIWorkflowRouter(_factory).router)

    response = TestClient(app).post(
        "/run",
        json={
            "threadId": "t1",
            "runId": "r1",
            "state": {},
            "messages": [],
            "tools": [],
            "context": [],
            "forwardedProps": {},
        },
    )

    events = _sse_events(response.text)
    types = [event["type"] for event in events]
    assert types == [
        "RUN_STARTED",
        "ACTIVITY_SNAPSHOT",
        "ACTIVITY_DELTA",
        "RUN_FINISHED",
    ]

    snapshot, delta = events[1], events[2]
    assert snapshot["messageId"] == "a1"
    assert snapshot["activityType"] == "a2ui-surface"
    assert snapshot["content"] == {"a2ui_operations": OPERATIONS}
    assert delta["patch"] == [{"op": "add", "path": "/status", "value": "done"}]
