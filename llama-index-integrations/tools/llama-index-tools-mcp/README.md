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

## Helper Functions

This package also includes several helper functions for working with MCP Servers.

### `workflow_as_mcp`

This function converts a `Workflow` to an MCP app.

```python
from llama_index.core.workflow import (
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.tools.mcp import workflow_as_mcp


class RunEvent(StartEvent):
    msg: str


class InfoEvent(Event):
    msg: str


class LoudWorkflow(Workflow):
    """Useful for converting strings to uppercase and making them louder."""

    @step
    def step_one(self, ctx: Context, ev: RunEvent) -> StopEvent:
        ctx.write_event_to_stream(InfoEvent(msg="Hello, world!"))

        return StopEvent(result=ev.msg.upper() + "!")


workflow = LoudWorkflow()

mcp = workflow_as_mcp(workflow, start_event_model=RunEvent)
```

Then, you can launch the MCP server (assuming you have the `mcp[cli]` extra installed):

```bash
mcp dev script.py
```

### `get_tools_from_mcp_url` / `aget_tools_from_mcp_url`

This function get a list of `FunctionTool`s from an MCP server or command.

```python
from llama_index.tools.mcp import (
    get_tools_from_mcp_url,
    aget_tools_from_mcp_url,
)

tools = get_tools_from_mcp_url("http://127.0.0.1:8000/sse")

# async
tools = await get_tools_from_mcp_url("http://127.0.0.1:8000/sse")
```

## Notebook Example

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-mcp/examples/mcp.ipynb).

This tool is designed to be used as a way to call the tools provided by MCP Servers.
