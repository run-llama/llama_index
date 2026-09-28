---
title: Using MCP Tools with LlamaIndex
---

LlamaIndex provides robust support for consuming MCP servers through the `llama-index-tools-mcp` package.

## Installation

```bash
pip install llama-index-tools-mcp
```

## Basic Usage

The most common usage will be converting an MCP server into a list of `llama-index` tool definitions:

```python
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# Connect to MCP server
mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
mcp_tool_spec = McpToolSpec(client=mcp_client)

# Get tools
tools = await mcp_tool_spec.to_tool_list_async()
```

## Using with Agents

Once you have a list of tools, you can plug this into any existing agent:

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

agent = FunctionAgent(
    tools=tools,
    llm=OpenAI(model="gpt-5-mini"),
    system_prompt="You are a helpful assistant.",
)

response = await agent.run("Your query here")
```

You can read more about agents in the [Agents Guide](/python/framework/understanding/agent).

## Connection Types

The `BasicMCPClient` supports multiple transport methods:

```python
# Server-Sent Events
sse_client = BasicMCPClient("https://example.com/sse")

# Streamable HTTP
http_client = BasicMCPClient("https://example.com/mcp")

# Local process
local_client = BasicMCPClient("python", args=["server.py"])
```

## ScreenContextAgent connector

[ScreenContextAgent](https://github.com/ikeikeikeda66/screen-context-agent) is a
local, encrypted screen-history MCP server. Its search results are OCR
observations, not instructions. LlamaIndex exposes a small wrapper that keeps
the server local, allows only `search_screen_history`, and requires an
explicit time window:

```bash
pip install llama-index-tools-mcp
```

```python
from datetime import datetime, timezone

from llama_index.tools.mcp import BasicMCPClient, ScreenContextConnector

client = BasicMCPClient(
    "screen-context",
    args=["serve", "--profile", "standard", "--transport", "stdio"],
    env={"SCREEN_CONTEXT_CLIENT_TOKEN": "<token from screen-context mcp-config>"},
)
connector = ScreenContextConnector(client)
tools = await connector.as_tool_spec().to_tool_list_async()
matches = await connector.search(
    "build error",
    start=datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc),
    end=datetime(2026, 9, 28, 10, 0, tzinfo=timezone.utc),
)
```

The connector filters the upper bound locally, preserves `frame_id`, timestamp
and source-app metadata, and labels every excerpt as untrusted observed data.
It never captures, writes or uploads screen history.
