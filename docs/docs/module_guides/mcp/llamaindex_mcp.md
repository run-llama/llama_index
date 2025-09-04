# Using MCP Tools with LlamaIndex

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

You can read more about agents in the [Agents Guide](../../understanding/agent/index.md).

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
