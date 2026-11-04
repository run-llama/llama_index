import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from httpx2 import AsyncClient
from mcp import types
from mcp.client.session import ClientSession

from llama_index.tools.mcp import BasicMCPClient, McpToolSpec


@pytest.fixture(params=["basic", "session"])
def paginated_client(request, mocker):
    session = MagicMock(spec=ClientSession)
    session.list_tools = AsyncMock()
    if request.param == "session":
        return session, session

    client = BasicMCPClient("unused", http_client=MagicMock(spec=AsyncClient))

    @asynccontextmanager
    async def run_session():
        yield session

    mocker.patch.object(client, "_run_session", side_effect=run_session)
    return client, session


def tool(name):
    return types.Tool(
        name=name,
        description=f"Call {name}",
        input_schema={
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}},
            "required": ["ticket_id"],
        },
    )


@pytest.mark.asyncio
async def test_tool_spec_discovers_all_pages(paginated_client):
    client, session = paginated_client
    first = types.ListToolsResult(tools=[tool("search_tickets")], next_cursor="")
    middle = types.ListToolsResult(tools=[], next_cursor="opaque:/+=next")
    last = types.ListToolsResult(tools=[tool("get_ticket")])
    session.list_tools.side_effect = [first, middle, last]

    tools = await McpToolSpec(client).to_tool_list_async()

    assert [item.metadata.name for item in tools] == ["search_tickets", "get_ticket"]
    parameters = tools[1].metadata.get_parameters_dict()
    assert parameters["properties"]["ticket_id"]["type"] == "string"
    assert parameters["required"] == ["ticket_id"]
    assert session.list_tools.await_args_list == [
        call(),
        call(params=types.PaginatedRequestParams(cursor="")),
        call(params=types.PaginatedRequestParams(cursor="opaque:/+=next")),
    ]
    assert first.tools == [tool("search_tickets")]
    assert first.next_cursor == ""
    if isinstance(client, BasicMCPClient):
        client._run_session.assert_called_once()


@pytest.mark.asyncio
async def test_tool_spec_filters_after_pagination(paginated_client):
    client, session = paginated_client
    session.list_tools.side_effect = [
        types.ListToolsResult(tools=[tool("search_tickets")], next_cursor="next"),
        types.ListToolsResult(tools=[tool("get_ticket")]),
    ]

    tools = await McpToolSpec(client, allowed_tools=["get_ticket"]).to_tool_list_async()

    assert [item.metadata.name for item in tools] == ["get_ticket"]


@pytest.mark.asyncio
async def test_tool_spec_empty_listing(paginated_client):
    client, session = paginated_client
    session.list_tools.return_value = types.ListToolsResult(tools=[])

    assert await McpToolSpec(client).to_tool_list_async() == []
    session.list_tools.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [RuntimeError("page failed"), asyncio.CancelledError()]
)
async def test_tool_spec_does_not_return_partial_listing(paginated_client, error):
    client, session = paginated_client
    session.list_tools.side_effect = [
        types.ListToolsResult(tools=[tool("search_tickets")], next_cursor="next"),
        error,
    ]

    with pytest.raises(type(error)):
        await McpToolSpec(client).to_tool_list_async()


@pytest.mark.asyncio
async def test_basic_client_returns_complete_result(mocker):
    client = BasicMCPClient("unused", http_client=MagicMock(spec=AsyncClient))
    session = MagicMock(spec=ClientSession)
    session.list_tools = AsyncMock(
        side_effect=[
            types.ListToolsResult(
                tools=[tool("search_tickets")],
                next_cursor="next",
                _meta={"source": "test"},
            ),
            types.ListToolsResult(tools=[tool("get_ticket")]),
        ]
    )

    @asynccontextmanager
    async def run_session():
        yield session

    mocker.patch.object(client, "_run_session", side_effect=run_session)

    result = await client.list_tools()

    assert [item.name for item in result.tools] == ["search_tickets", "get_ticket"]
    assert result.next_cursor is None
    assert result.meta == {"source": "test"}
    client._run_session.assert_called_once()
