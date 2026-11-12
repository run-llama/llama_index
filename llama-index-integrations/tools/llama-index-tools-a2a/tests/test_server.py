"""Tests for the A2A server (inbound workflow-as-a2a)."""

import pytest
from httpx import ASGITransport, AsyncClient
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step
from llama_index.tools.a2a import workflow_as_a2a


class SimpleStart(StartEvent):
    query: str


class SimpleWorkflow(Workflow):
    @step
    async def run(self, start_event: SimpleStart) -> StopEvent:
        return StopEvent(result=f"Response to: {start_event.query}")


class MultiStepWorkflow(Workflow):
    @step
    async def process(self, start_event: SimpleStart) -> StopEvent:
        intermediate = f"Processed: {start_event.query}"
        return StopEvent(result=intermediate)


@pytest.mark.asyncio
async def test_agent_card():
    """Test that the AgentCard endpoint returns correct metadata."""
    app = workflow_as_a2a(SimpleWorkflow())
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/.well-known/agent-card.json")
        assert r.status_code == 200
        card = r.json()
        assert card["name"] == "SimpleWorkflow"
        assert card["version"] == "1.0.0"
        assert len(card["skills"]) == 1
        assert card["skills"][0]["id"] == "simpleworkflow"
        assert card["capabilities"]["streaming"] is True


@pytest.mark.asyncio
async def test_message_send_success():
    """Test successful message/send flow."""
    app = workflow_as_a2a(SimpleWorkflow())
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "msg-1",
                        "role": "user",
                        "parts": [{"kind": "text", "text": '{"query": "hello"}'}],
                    }
                },
                "id": 1,
            },
        )
        assert resp.status_code == 200
        result = resp.json()
        assert "result" in result
        task = result["result"]
        assert task["status"]["state"] == "completed"
        assert task["artifacts"][0]["parts"][0]["text"] == '{"result":"Response to: hello"}'


@pytest.mark.asyncio
async def test_tasks_get():
    """Test retrieving a completed task via tasks/get."""
    app = workflow_as_a2a(SimpleWorkflow())
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Send a message first
        send_resp = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "msg-2",
                        "role": "user",
                        "parts": [{"kind": "text", "text": '{"query": "test"}'}],
                    }
                },
                "id": 2,
            },
        )
        task_id = send_resp.json()["result"]["id"]

        # Get the task
        get_resp = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "method": "tasks/get",
                "params": {"id": task_id},
                "id": 3,
            },
        )
        assert get_resp.status_code == 200
        task = get_resp.json()["result"]
        assert task["status"]["state"] == "completed"


@pytest.mark.asyncio
async def test_empty_input_fails_gracefully():
    """Test that empty input results in a failed status."""
    app = workflow_as_a2a(SimpleWorkflow())
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "msg-empty",
                        "role": "user",
                        "parts": [{"kind": "text", "text": ""}],
                    }
                },
                "id": 4,
            },
        )
        assert resp.status_code == 200
        result = resp.json()
        # Should fail gracefully
        assert "error" in result or result.get("result", {}).get("status", {}).get("state") == "failed"


@pytest.mark.asyncio
async def test_workflow_with_string_input():
    """Test workflow accepting string input (single-field StartEvent)."""

    class StringStart(StartEvent):
        text: str

    class StringWorkflow(Workflow):
        @step
        async def run(self, start_event: StringStart) -> StopEvent:
            return StopEvent(result=f"Got: {start_event.text}")

    app = workflow_as_a2a(StringWorkflow())
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "msg-str",
                        "role": "user",
                        "parts": [{"kind": "text", "text": "plain text input"}],
                    }
                },
                "id": 5,
            },
        )
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["status"]["state"] == "completed"
