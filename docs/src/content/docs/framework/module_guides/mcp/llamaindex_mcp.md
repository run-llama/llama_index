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

## LlamaIndex-hosted MCP servers

You can point `BasicMCPClient` at LlamaIndex's own MCP servers. The [LlamaParse MCP
server](https://developers.llamaindex.ai/llamaparse/for-agents/mcp/) exposes document parsing,
classification, extraction, splitting, and indexing as tools, and authenticates with OAuth:

```python
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

client = BasicMCPClient.with_oauth(
    "https://mcp.llamaindex.ai/mcp",
    client_name="My App",
    redirect_uris=["http://localhost:3000/callback"],
    # Function to handle the redirect URL (e.g., open a browser)
    redirect_handler=lambda url: print(f"Please visit: {url}"),
    # Function to get the authorization code from the user
    callback_handler=lambda: (input("Enter the code: "), None),
)

tools = await McpToolSpec(client=client).to_tool_list_async()
```

By default `with_oauth` stores tokens in memory, so they are lost on restart. See the
[`llama-index-tools-mcp`
README](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/tools/llama-index-tools-mcp)
for implementing persistent token storage.

Narrower servers are available for individual products (for example
`https://mcp.llamaindex.ai/extract/mcp`), which gives agents a smaller tool list to choose from.
