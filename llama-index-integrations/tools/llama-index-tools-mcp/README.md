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
    # Optional: Include resources in the tool list
    # include_resources=True,
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

## MCP Client Usage

The `BasicMCPClient` provides comprehensive access to MCP server capabilities beyond just tools.

### Basic Client Operations

```python
from llama_index.tools.mcp import BasicMCPClient

# Connect to an MCP server using different transports
http_client = BasicMCPClient("https://example.com/mcp")  # Streamable HTTP
sse_client = BasicMCPClient("https://example.com/sse")  # Server-Sent Events
local_client = BasicMCPClient("python", args=["server.py"])  # stdio

# List available tools
tools = await http_client.list_tools()

# Call a tool
result = await http_client.call_tool("calculate", {"x": 5, "y": 10})

# List available resources
resources = await http_client.list_resources()

# Read a resource
content, mime_type = await http_client.read_resource("config://app")

# List available prompts
prompts = await http_client.list_prompts()

# Get a prompt
prompt_result = await http_client.get_prompt("greet", {"name": "World"})
```

### OAuth Authentication

The client supports OAuth 2.0 authentication for connecting to protected MCP servers:

```python
from llama_index.tools.mcp import BasicMCPClient

# Simple authentication with in-memory token storage
client = BasicMCPClient.with_oauth(
    "https://api.example.com/mcp",
    client_name="My App",
    redirect_uris=["http://localhost:3000/callback"],
    # Function to handle the redirect URL (e.g., open a browser)
    redirect_handler=lambda url: print(f"Please visit: {url}"),
    # Function to get the authorization code from the user
    callback_handler=lambda: (input("Enter the code: "), None),
)

# Use the authenticated client
tools = await client.list_tools()
```

For production use, you can implement a custom token storage:

```python
from llama_index.tools.mcp import BasicMCPClient
from mcp.client.auth import TokenStorage
from mcp.shared.auth import OAuthToken, OAuthClientInformationFull
import json
import os


class FileTokenStorage(TokenStorage):
    """Store OAuth tokens in a file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._client_info: Optional[OAuthClientInformationFull] = None

    async def get_tokens(self):
        if not os.path.exists(self.file_path):
            return None
        with open(self.file_path, "r") as f:
            data = json.load(f)
            return OAuthToken(**data.get("tokens", {}))

    async def set_tokens(self, tokens):
        data = {}
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                data = json.load(f)
        data["tokens"] = tokens.__dict__
        with open(self.file_path, "w") as f:
            json.dump(data, f)

    async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
        """Get the stored client information."""
        return self._client_info

    async def set_client_info(
        self, client_info: OAuthClientInformationFull
    ) -> None:
        """Store client information."""
        self._client_info = client_info


# Use custom storage
client = BasicMCPClient.with_oauth(
    "https://api.example.com/mcp",
    client_name="My App",
    redirect_uris=["http://localhost:3000/callback"],
    redirect_handler=lambda url: print(f"Please visit: {url}"),
    callback_handler=lambda: (input("Enter the code: "), None),
    token_storage=FileTokenStorage("tokens.json"),
)
```

## Notebook Example

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-mcp/examples/mcp.ipynb).

This tool is designed to be used as a way to call the tools provided by MCP Servers.
