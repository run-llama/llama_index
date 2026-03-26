"""
Integration tests for AgentCoreRuntime.

Tests the full HTTP round-trip through BedrockAgentCoreApp using httpx's ASGI
transport (no real server needed, no AWS credentials needed). Uses
MockFunctionCallingLLM from llama-index-core which echoes user messages.

Validates:
  - Non-streaming JSON response via POST /invocations
  - Streaming SSE response via POST /invocations
  - GET /ping health check
  - 400 error for missing prompt
  - Session ID header propagation to memory
  - Various payload key formats (prompt, message, input)

Run:
    uv run pytest tests/test_runtime_e2e.py -v
"""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from llama_index.core.agent.workflow import FunctionAgent

try:
    from llama_index.core.llms.mock import MockFunctionCallingLLM
except ImportError:
    MockFunctionCallingLLM = None

from llama_index.tools.aws_bedrock_agentcore.runtime.base import AgentCoreRuntime

pytestmark = pytest.mark.skipif(
    MockFunctionCallingLLM is None,
    reason="MockFunctionCallingLLM not available in this llama-index-core version",
)

SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"


@pytest.fixture
def agent():
    llm = MockFunctionCallingLLM()
    return FunctionAgent(llm=llm, tools=[])


@pytest.fixture
def non_streaming_client(agent):
    runtime = AgentCoreRuntime(agent=agent, stream=False)
    transport = httpx.ASGITransport(app=runtime.app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture
def streaming_client(agent):
    runtime = AgentCoreRuntime(agent=agent, stream=True)
    transport = httpx.ASGITransport(app=runtime.app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


class TestPing:
    @pytest.mark.asyncio
    async def test_ping(self, non_streaming_client):
        resp = await non_streaming_client.get("/ping")
        assert resp.status_code == 200


class TestNonStreaming:
    @pytest.mark.asyncio
    async def test_prompt_key(self, non_streaming_client):
        resp = await non_streaming_client.post(
            "/invocations", json={"prompt": "hello world"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "response" in body
        # MockFunctionCallingLLM echoes user message
        assert "hello world" in body["response"]

    @pytest.mark.asyncio
    async def test_message_key(self, non_streaming_client):
        resp = await non_streaming_client.post(
            "/invocations", json={"message": "test message"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "test message" in body["response"]

    @pytest.mark.asyncio
    async def test_input_key(self, non_streaming_client):
        resp = await non_streaming_client.post(
            "/invocations", json={"input": "test input"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "test input" in body["response"]

    @pytest.mark.asyncio
    async def test_missing_prompt_returns_400(self, non_streaming_client):
        resp = await non_streaming_client.post("/invocations", json={"foo": "bar"})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_payload_returns_400(self, non_streaming_client):
        resp = await non_streaming_client.post("/invocations", json={})
        assert resp.status_code == 400


class TestStreaming:
    @pytest.mark.asyncio
    async def test_streaming_returns_sse(self, streaming_client):
        resp = await streaming_client.post(
            "/invocations", json={"prompt": "stream test"}
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        events = _parse_sse(resp.text)
        assert len(events) > 0

        event_types = [e.get("event") for e in events]
        assert "done" in event_types

        done_events = [e for e in events if e.get("event") == "done"]
        assert len(done_events) == 1
        assert "stream test" in done_events[0]["response"]

    @pytest.mark.asyncio
    async def test_streaming_agent_stream_events(self, streaming_client):
        resp = await streaming_client.post(
            "/invocations", json={"prompt": "token test"}
        )
        events = _parse_sse(resp.text)

        stream_events = [e for e in events if e.get("event") == "agent_stream"]
        assert len(stream_events) > 0
        for se in stream_events:
            assert "delta" in se
            assert "response" in se

    @pytest.mark.asyncio
    async def test_streaming_missing_prompt_returns_error(self, streaming_client):
        resp = await streaming_client.post("/invocations", json={"foo": "bar"})
        # For streaming, BedrockAgentCoreApp returns 200 (SSE connection open)
        # but includes an error event in the stream. The error is raised inside
        # the generator after the response headers are sent.
        assert resp.status_code == 200
        # The response should contain an error indicator in the SSE stream
        assert "error" in resp.text.lower() or "HTTPException" in resp.text


class TestSessionIdPropagation:
    @pytest.mark.asyncio
    async def test_session_header_reaches_handler(self):
        """Verify that session ID from header is accessible in the handler."""
        captured_contexts = []

        agent = FunctionAgent(llm=MockFunctionCallingLLM(), tools=[])
        runtime = AgentCoreRuntime(agent=agent, stream=False)

        original_get_memory = runtime._get_memory

        def capturing_get_memory(context=None):
            if context:
                captured_contexts.append(context)
            return original_get_memory(context)

        runtime._get_memory = capturing_get_memory

        transport = httpx.ASGITransport(app=runtime.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            resp = await client.post(
                "/invocations",
                json={"prompt": "session test"},
                headers={SESSION_HEADER: "test-session-abc"},
            )

        assert resp.status_code == 200
        assert len(captured_contexts) == 1
        assert captured_contexts[0].session_id == "test-session-abc"

    @pytest.mark.asyncio
    async def test_session_id_wired_to_memory(self):
        """Verify session ID from header is set on a copy of memory, not the original."""
        mock_memory = AsyncMock()
        mock_memory._context = MagicMock()
        mock_memory._context.session_id = "old-session"
        mock_memory.aput.return_value = None
        mock_memory.aget.return_value = []
        mock_memory.aget_all.return_value = []
        mock_memory.aput_messages.return_value = None

        agent = FunctionAgent(llm=MockFunctionCallingLLM(), tools=[])
        runtime = AgentCoreRuntime(agent=agent, stream=False, memory=mock_memory)

        # Capture the memory returned by _get_memory
        captured_memories = []
        original_get_memory = runtime._get_memory

        def capturing_get_memory(context=None):
            result = original_get_memory(context)
            captured_memories.append(result)
            return result

        runtime._get_memory = capturing_get_memory

        transport = httpx.ASGITransport(app=runtime.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            resp = await client.post(
                "/invocations",
                json={"prompt": "memory test"},
                headers={SESSION_HEADER: "new-session-xyz"},
            )

        assert resp.status_code == 200
        assert len(captured_memories) == 1
        # The returned memory should have the new session ID
        assert captured_memories[0]._context.session_id == "new-session-xyz"
        # Original memory should be unchanged (no race condition)
        assert mock_memory._context.session_id == "old-session"


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text into list of JSON event dicts."""
    events = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            try:
                events.append(json.loads(data))
            except json.JSONDecodeError:
                pass
    return events
