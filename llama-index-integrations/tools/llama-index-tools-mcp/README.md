# MCP ToolSpec

This tool connects to MCP Servers and allows an Agent to call the tools provided by MCP Servers.

This idea is migrated from [Integrate MCP Tools into LlamaIndex](https://psiace.me/posts/integrate-mcp-tools-into-llamaindex/).

## Installation

```bash
pip install llama-index-tools-mcp
```

## Usage

Usage is as simple as connecting to an MCP Server and getting the tools.

```python
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# We consider there is a mcp server running on 127.0.0.1:8000, or you can use the mcp client to connect to your own mcp server.
mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
mcp_tool_spec = McpToolSpec(
    client=mcp_client,
    # Optional: Filter the tools by name
    # allowed_tools=["tool1", "tool2"],
)

# sync
tools = mcp_tool_spec.to_tool_list()

# async
tools = await mcp_tool_spec.to_tool_list_async()
```

Then you can use the tools in your agent!

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

agent = FunctionAgent(
    name="Agent",
    description="Some description",
    llm=OpenAI(model="gpt-4o"),
    tools=tools,
    system_prompt="You are a helpful assistant.",
)

resp = await agent.run("What is the weather in Tokyo?")
```

## Notebook Example

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-mcp/examples/mcp.ipynb).

This tool is designed to be used as a way to call the tools provided by MCP Servers.
