"""
Unit tests for MCP resource-template handling in McpToolSpec.

These tests use lightweight fakes and do NOT require a running MCP server,
so they run quickly in isolation and cover the code paths from issue #23001.
"""

from types import SimpleNamespace
from typing import List

import pytest

from llama_index.tools.mcp.base import McpToolSpec, _extract_template_vars


class _FakeClient:
    """
    Minimal ClientSession stand-in for exercising McpToolSpec.

    Only exposes the methods McpToolSpec touches when building the tool list.
    """

    def __init__(
        self,
        static_resources: List[SimpleNamespace] = (),
        resource_templates: List[SimpleNamespace] = (),
    ) -> None:
        self.static_resources = list(static_resources)
        self.resource_templates = list(resource_templates)
        self.read_calls: List[str] = []

    async def list_tools(self):
        return SimpleNamespace(tools=[])

    async def list_resources(self):
        return SimpleNamespace(resources=self.static_resources)

    async def list_resource_templates(self):
        return SimpleNamespace(resource_templates=self.resource_templates)

    async def read_resource(self, uri):
        self.read_calls.append(uri)
        return f"content-for:{uri}"


def _static(name: str, uri: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, uri=uri, description=f"static {name}")


def _template(name: str, uri_template: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        uri_template=uri_template,
        description=f"template {name}",
    )


def test_extract_template_vars_orders_and_dedupes():
    assert _extract_template_vars("repo://{owner}/{name}/README.md") == [
        "owner",
        "name",
    ]
    assert _extract_template_vars("path://{a}/{b}/{a}") == ["a", "b"]
    assert _extract_template_vars("no-vars://static") == []


@pytest.mark.asyncio
async def test_static_resource_still_zero_argument():
    client = _FakeClient(static_resources=[_static("cfg", "config://app")])
    spec = McpToolSpec(client, include_resources=True)

    tools = await spec.to_tool_list_async()

    assert len(tools) == 1
    tool = tools[0]
    assert tool.metadata.name == "cfg"
    # No template variables -> schema has no required fields.
    props = tool.metadata.fn_schema.model_json_schema().get("properties", {})
    assert props == {}

    result = await tool.acall()
    assert client.read_calls == ["config://app"]
    assert "config://app" in result.content


@pytest.mark.asyncio
async def test_resource_template_exposes_variables_and_expands_uri():
    client = _FakeClient(
        resource_templates=[_template("repo_readme", "repo://{owner}/{name}/README.md")]
    )
    spec = McpToolSpec(client, include_resources=True)

    tools = await spec.to_tool_list_async()

    assert len(tools) == 1
    tool = tools[0]
    assert tool.metadata.name == "repo_readme"

    schema = tool.metadata.fn_schema.model_json_schema()
    assert set(schema["properties"].keys()) == {"owner", "name"}
    assert set(schema.get("required", [])) == {"owner", "name"}

    await tool.acall(owner="openai", name="openai-agents-python")
    assert client.read_calls == ["repo://openai/openai-agents-python/README.md"]


@pytest.mark.asyncio
async def test_resource_template_with_repeated_variable():
    client = _FakeClient(resource_templates=[_template("dup", "x://{a}/{b}/{a}")])
    spec = McpToolSpec(client, include_resources=True)

    (tool,) = await spec.to_tool_list_async()
    schema = tool.metadata.fn_schema.model_json_schema()
    assert set(schema["properties"].keys()) == {"a", "b"}

    await tool.acall(a="1", b="2")
    assert client.read_calls == ["x://1/2/1"]


@pytest.mark.asyncio
async def test_mixed_static_and_template_do_not_leak_uri_between_iterations():
    """
    Regression: previously a template resource fell through to the
    prior iteration's `uri`, so the template tool called read_resource with
    the last static resource's URI.
    """
    client = _FakeClient(
        static_resources=[_static("cfg", "config://app")],
        resource_templates=[_template("user_profile", "users://{user_id}/profile")],
    )
    spec = McpToolSpec(client, include_resources=True)

    tools = await spec.to_tool_list_async()
    tools_by_name = {t.metadata.name: t for t in tools}

    assert set(tools_by_name) == {"cfg", "user_profile"}

    await tools_by_name["user_profile"].acall(user_id="42")
    assert client.read_calls == ["users://42/profile"]
