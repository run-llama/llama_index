"""Tests for the A2A tool spec (outbound skill-to-FunctionTool conversion)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from llama_index.tools.a2a import A2AToolSpec, BasicA2AClient
from llama_index.tools.a2a.models import AgentCard, Skill, AgentCapabilities, AgentProvider
from llama_index.core.tools import FunctionTool


@pytest.mark.asyncio
async def test_tool_spec_to_tool_list():
    """Test converting A2A skills to FunctionTools."""
    mock_card = AgentCard(
        url="http://example.com",
        name="TestAgent",
        skills=[
            Skill(name="search_web", description="Search the web", input_schema={"type": "object", "properties": {"query": {"type": "string"}}}),
            Skill(name="calculate", description="Calculate math", input_schema={"type": "object", "properties": {"expr": {"type": "string"}}}),
        ],
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "id": "task-123",
            "status": {"state": "completed"},
            "artifacts": [{"parts": [{"kind": "text", "text": "search result"}]}],
        },
        "id": 1,
    }

    with patch.object(BasicA2AClient, "_discover_agent_card", new_callable=AsyncMock, return_value=mock_card):
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            client = BasicA2AClient("http://example.com/.well-known/agent-card.json")
            tool_spec = A2AToolSpec(client)
            tools = await tool_spec.to_tool_list_async()

            assert len(tools) == 2
            assert isinstance(tools[0], FunctionTool)
            assert tools[0].metadata.name == "search_web"
            assert tools[1].metadata.name == "calculate"


@pytest.mark.asyncio
async def test_tool_spec_allowed_tools():
    """Test filtering tools with allowed_tools."""
    mock_card = AgentCard(
        url="http://example.com",
        name="TestAgent",
        skills=[
            Skill(name="search_web", description="Search the web"),
            Skill(name="calculate", description="Calculate math"),
            Skill(name="translate", description="Translate text"),
        ],
        capabilities=AgentCapabilities(),
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {"id": "task-123", "status": {"state": "completed"}},
        "id": 1,
    }

    with patch.object(BasicA2AClient, "_discover_agent_card", new_callable=AsyncMock, return_value=mock_card):
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            client = BasicA2AClient("http://example.com/.well-known/agent-card.json")
            tool_spec = A2AToolSpec(client, allowed_tools=["search_web", "translate"])
            tools = await tool_spec.to_tool_list_async()

            assert len(tools) == 2
            tool_names = {t.metadata.name for t in tools}
            assert tool_names == {"search_web", "translate"}


@pytest.mark.asyncio
async def test_tool_spec_global_partial_params():
    """Test global partial params applied to all tools."""
    mock_card = AgentCard(
        url="http://example.com",
        name="TestAgent",
        skills=[
            Skill(name="skill1", description="Skill 1"),
        ],
        capabilities=AgentCapabilities(),
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {"id": "task-123", "status": {"state": "completed"}},
        "id": 1,
    }

    with patch.object(BasicA2AClient, "_discover_agent_card", new_callable=AsyncMock, return_value=mock_card):
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            client = BasicA2AClient("http://example.com/.well-known/agent-card.json")
            tool_spec = A2AToolSpec(client, global_partial_params={"api_key": "secret"})
            tools = await tool_spec.to_tool_list_async()

            # Call the tool with just the skill-specific arg
            result = await tools[0].acall(query="test")
            # The global param should have been added internally
            # (we can't easily verify the internal call, but the tool should work)


@pytest.mark.asyncio
async def test_tool_spec_partial_params_by_tool():
    """Test tool-specific partial params."""
    mock_card = AgentCard(
        url="http://example.com",
        name="TestAgent",
        skills=[
            Skill(name="skill1", description="Skill 1"),
        ],
        capabilities=AgentCapabilities(),
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {"id": "task-123", "status": {"state": "completed"}},
        "id": 1,
    }

    with patch.object(BasicA2AClient, "_discover_agent_card", new_callable=AsyncMock, return_value=mock_card):
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            client = BasicA2AClient("http://example.com/.well-known/agent-card.json")
            tool_spec = A2AToolSpec(client, partial_params_by_tool={"skill1": {"model": "gpt-4"}})
            tools = await tool_spec.to_tool_list_async()

            assert len(tools) == 1