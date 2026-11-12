"""Tests for the A2A outbound client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from llama_index.tools.a2a.client import BasicA2AClient
from llama_index.tools.a2a.models import AgentCard, Skill, Part


@pytest.mark.asyncio
async def test_list_skills():
    """Test listing skills from agent card."""
    mock_card = AgentCard(
        url="http://example.com",
        name="TestAgent",
        skills=[
            Skill(name="skill1", description="First skill"),
            Skill(name="skill2", description="Second skill"),
        ],
    )

    with patch.object(BasicA2AClient, "_discover_agent_card", new_callable=AsyncMock, return_value=mock_card):
        client = BasicA2AClient("http://example.com/.well-known/agent-card.json")
        skills = await client.list_skills()
        assert len(skills) == 2
        assert skills[0].name == "skill1"
        assert skills[1].name == "skill2"


@pytest.mark.asyncio
async def test_call_skill():
    """Test calling a skill through tasks/send."""
    mock_card = AgentCard(
        url="http://example.com",
        name="TestAgent",
        skills=[Skill(name="my-skill", description="A skill")],
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "id": "task-123",
            "contextId": "ctx-456",
            "status": {"state": "completed"},
            "artifacts": [{"parts": [{"kind": "text", "text": "result"}]}],
        },
        "id": 1,
    }

    with patch.object(BasicA2AClient, "_discover_agent_card", new_callable=AsyncMock, return_value=mock_card):
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            client = BasicA2AClient("http://example.com/.well-known/agent-card.json")
            result = await client.call_skill("my-skill", arguments={"key": "value"})
            assert result.task_id == "task-123"


@pytest.mark.asyncio
async def test_send_message():
    """Test sending a natural language message."""
    mock_card = AgentCard(
        url="http://example.com",
        name="TestAgent",
        skills=[],
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "id": "task-789",
            "status": {"state": "completed"},
        },
        "id": 1,
    }

    with patch.object(BasicA2AClient, "_discover_agent_card", new_callable=AsyncMock, return_value=mock_card):
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            client = BasicA2AClient("http://example.com/.well-known/agent-card.json")
            result = await client.send_message("Hello, agent!")
            assert result.task_id == "task-789"


@pytest.mark.asyncio
async def test_close():
    """Test closing the HTTP client."""
    mock_client = AsyncMock()
    client = BasicA2AClient("http://example.com")
    client._http_client = mock_client
    await client.close()
    mock_client.aclose.assert_called_once()
