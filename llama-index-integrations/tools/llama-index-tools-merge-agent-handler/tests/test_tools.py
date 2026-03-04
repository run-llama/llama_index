from __future__ import annotations

import json

import httpx
import pytest
import respx
from llama_index.core.tools.function_tool import FunctionTool

from llama_index.tools.merge_agent_handler import MergeAgentHandlerToolSpec


@pytest.fixture
def tool_spec() -> MergeAgentHandlerToolSpec:
    spec = MergeAgentHandlerToolSpec(
        api_key="test-api-key",
        tool_pack_id="tp_default",
        registered_user_id="ru_default",
    )
    yield spec
    spec.close()


def test_spec_functions_are_expected() -> None:
    assert MergeAgentHandlerToolSpec.spec_functions == [
        "list_tool_packs",
        "list_registered_users",
        "list_tools",
        "call_tool",
    ]


def test_init_stores_credentials_and_defaults(tool_spec: MergeAgentHandlerToolSpec) -> None:
    assert tool_spec.api_key == "test-api-key"
    assert tool_spec.tool_pack_id == "tp_default"
    assert tool_spec.registered_user_id == "ru_default"
    assert tool_spec.environment == "production"


@respx.mock
def test_fetch_all_pages_handles_pagination(tool_spec: MergeAgentHandlerToolSpec) -> None:
    def _paged_response(request: httpx.Request) -> httpx.Response:
        page = request.url.params.get("page")
        if page == "1":
            return httpx.Response(
                200,
                json={
                    "results": [{"id": "tp_1", "name": "Finance Pack", "connectors": []}],
                    "next": "https://ah-api.merge.dev/api/v1/tool-packs/?page=2",
                },
            )
        if page == "2":
            return httpx.Response(
                200,
                json={
                    "results": [{"id": "tp_2", "name": "Support Pack", "connectors": []}],
                    "next": None,
                },
            )
        return httpx.Response(404, json={"detail": "unexpected page"})

    route = respx.get("https://ah-api.merge.dev/api/v1/tool-packs/").mock(side_effect=_paged_response)

    result = tool_spec._fetch_all_pages("/tool-packs/")
    assert route.call_count == 2
    assert [item["id"] for item in result] == ["tp_1", "tp_2"]


@respx.mock
def test_list_tool_packs_returns_expected_json(tool_spec: MergeAgentHandlerToolSpec) -> None:
    respx.get("https://ah-api.merge.dev/api/v1/tool-packs/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "tp_1",
                        "name": "Support Pack",
                        "description": "Ticketing tools",
                        "connectors": [{"name": "Jira", "slug": "jira", "id": "ignored"}],
                    }
                ],
                "next": None,
            },
        )
    )

    result = json.loads(tool_spec.list_tool_packs())
    assert result == [
        {
            "id": "tp_1",
            "name": "Support Pack",
            "description": "Ticketing tools",
            "connectors": [{"name": "Jira", "slug": "jira"}],
        }
    ]


@respx.mock
def test_list_registered_users_applies_environment_filter(tool_spec: MergeAgentHandlerToolSpec) -> None:
    route = respx.get("https://ah-api.merge.dev/api/v1/registered-users").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "ru_1",
                        "origin_user_name": "Alex",
                        "authenticated_connectors": ["jira"],
                        "is_test": True,
                    }
                ],
                "next": None,
            },
        )
    )

    result = json.loads(tool_spec.list_registered_users(environment="test"))
    assert route.calls[0].request.url.params["is_test"] == "true"
    assert result == [
        {
            "id": "ru_1",
            "origin_user_name": "Alex",
            "authenticated_connectors": ["jira"],
        }
    ]


@respx.mock
def test_list_tools_sends_expected_mcp_request(tool_spec: MergeAgentHandlerToolSpec) -> None:
    route = respx.post(
        "https://ah-api.merge.dev/api/v1/tool-packs/tp_default/registered-users/ru_default/mcp"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "tools": [
                        {
                            "name": "jira_list_tickets",
                            "description": "List Jira tickets",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                },
            },
        )
    )

    result = json.loads(tool_spec.list_tools())
    assert result == [{"name": "jira_list_tickets", "description": "List Jira tickets"}]

    payload = json.loads(route.calls[0].request.content.decode())
    assert payload["method"] == "tools/list"
    assert payload["params"] == {}


@respx.mock
def test_list_tools_uses_parameter_overrides(tool_spec: MergeAgentHandlerToolSpec) -> None:
    route = respx.post(
        "https://ah-api.merge.dev/api/v1/tool-packs/tp_override/registered-users/ru_override/mcp"
    ).mock(
        return_value=httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}},
        )
    )

    result = json.loads(tool_spec.list_tools(tool_pack_id="tp_override", registered_user_id="ru_override"))
    assert result == []
    assert route.call_count == 1


@respx.mock
def test_list_tools_raises_on_mcp_error(tool_spec: MergeAgentHandlerToolSpec) -> None:
    respx.post("https://ah-api.merge.dev/api/v1/tool-packs/tp_default/registered-users/ru_default/mcp").mock(
        return_value=httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": 1, "error": {"code": 500, "message": "boom"}},
        )
    )

    with pytest.raises(ValueError, match="MCP tools/list failed: boom"):
        tool_spec.list_tools()


@respx.mock
def test_call_tool_wraps_arguments_under_input(tool_spec: MergeAgentHandlerToolSpec) -> None:
    route = respx.post(
        "https://ah-api.merge.dev/api/v1/tool-packs/tp_default/registered-users/ru_default/mcp"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"content": [{"type": "text", "text": "ok"}], "isError": False},
            },
        )
    )

    result = tool_spec.call_tool("jira_list_tickets", arguments='{"limit": 5}')
    assert result == "ok"

    payload = json.loads(route.calls[0].request.content.decode())
    assert payload["method"] == "tools/call"
    assert payload["params"]["name"] == "jira_list_tickets"
    assert payload["params"]["arguments"] == {"input": {"limit": 5}}


@respx.mock
def test_call_tool_returns_error_string_on_mcp_error(tool_spec: MergeAgentHandlerToolSpec) -> None:
    respx.post("https://ah-api.merge.dev/api/v1/tool-packs/tp_default/registered-users/ru_default/mcp").mock(
        return_value=httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": 1, "error": {"code": 500, "message": "boom"}},
        )
    )

    result = tool_spec.call_tool("jira_list_tickets", arguments="{}")
    assert result == 'Tool "jira_list_tickets" returned error: boom'


@respx.mock
def test_call_tool_returns_error_string_on_is_error(tool_spec: MergeAgentHandlerToolSpec) -> None:
    respx.post("https://ah-api.merge.dev/api/v1/tool-packs/tp_default/registered-users/ru_default/mcp").mock(
        return_value=httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": "permission denied"}],
                },
            },
        )
    )

    result = tool_spec.call_tool("jira_list_tickets", arguments="{}")
    assert result == 'Tool "jira_list_tickets" failed: permission denied'


def test_call_tool_rejects_invalid_argument_json(tool_spec: MergeAgentHandlerToolSpec) -> None:
    result = tool_spec.call_tool("jira_list_tickets", arguments="{invalid")
    assert "arguments must be a valid JSON string representing an object" in result


def test_list_registered_users_rejects_invalid_environment(tool_spec: MergeAgentHandlerToolSpec) -> None:
    with pytest.raises(ValueError, match="environment must be either"):
        tool_spec.list_registered_users(environment="staging")


def test_to_tool_list_returns_four_function_tools(tool_spec: MergeAgentHandlerToolSpec) -> None:
    tools = tool_spec.to_tool_list()
    assert len(tools) == 4
    assert all(isinstance(tool, FunctionTool) for tool in tools)
