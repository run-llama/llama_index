# Hugging Face MCP Server Tool Integration

This package provides a LlamaIndex tool integration for the [Hugging Face MCP server](https://huggingface.co/docs/hub/en/hf-mcp-server). It enables LlamaIndex agents to search and explore Hugging Face Hub resources — models, datasets, Spaces, papers, and documentation — via the Model Context Protocol (MCP).

## Installation

```bash
pip install llama-index-tools-mcp-hf-server
```

## Available Tools

The Hugging Face MCP server exposes several built-in tools:

| Tool | Description |
|------|-------------|
| `hf_space_search` | Find AI apps/Spaces via natural language queries |
| `hf_paper_search` | Find ML research papers via natural language queries |
| `hf_model_search` | Search for ML models with filters (task, library, etc.) |
| `hf_dataset_search` | Search for datasets with filters (author, tags, etc.) |
| `hf_doc_search` | Search Hugging Face documentation using natural language |
| `hf_doc_fetch` | Fetch Hugging Face documentation content |
| `hf_repo_details` | Get detailed info about models, datasets, and Spaces |
| `hf_jobs` | Run, monitor, and schedule jobs on HF infrastructure |

## Quick Start

### Basic Usage

```python
from llama_index.tools.mcp_hf_server import HfMcpToolSpec

# Create tool spec (connects to HF MCP server automatically)
tool_spec = HfMcpToolSpec()

# Get all available tools
tools = await tool_spec.to_tool_list_async()
```

### With an Agent

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.mcp_hf_server import HfMcpToolSpec

tool_spec = HfMcpToolSpec()
tools = await tool_spec.to_tool_list_async()

agent = FunctionAgent(
    name="HFAgent",
    llm=OpenAI(model="gpt-4o"),
    tools=tools,
    system_prompt="You are a helpful assistant that can search the Hugging Face Hub.",
)

response = await agent.run("Find the top text generation models on Hugging Face")
print(response)
```

### Filter to Specific Tools

```python
from llama_index.tools.mcp_hf_server import HfMcpToolSpec

# Only expose model and dataset search
tool_spec = HfMcpToolSpec(
    allowed_tools=["hf_model_search", "hf_dataset_search"]
)
tools = await tool_spec.to_tool_list_async()
```

### Authenticated Access

```python
from llama_index.tools.mcp_hf_server import HfMcpToolSpec

# Use a Hugging Face API token for authenticated access
tool_spec = HfMcpToolSpec(hf_token="hf_your_token_here")
tools = await tool_spec.to_tool_list_async()
```

### Convenience Functions

```python
from llama_index.tools.mcp_hf_server import get_hf_tools, aget_hf_tools

# Synchronous
tools = get_hf_tools(allowed_tools=["hf_model_search"])

# Asynchronous
tools = await aget_hf_tools(allowed_tools=["hf_model_search"])
```

### Custom Client

```python
from llama_index.tools.mcp import BasicMCPClient
from llama_index.tools.mcp_hf_server import HfMcpToolSpec

# Use a custom MCP client (e.g., self-hosted HF MCP server)
client = BasicMCPClient("http://localhost:3000/mcp")
tool_spec = HfMcpToolSpec(client=client)
tools = await tool_spec.to_tool_list_async()
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `BasicMCPClient` | `None` | Custom MCP client. Auto-created if not provided. |
| `allowed_tools` | `List[str]` | `None` | Filter to specific tools by name. |
| `include_resources` | `bool` | `False` | Include MCP resources as tools. |
| `hf_token` | `str` | `None` | Hugging Face API token for authenticated access. |
| `use_auth_url` | `bool` | `True` | Use the authenticated MCP endpoint. |
| `timeout` | `int` | `60` | Connection timeout in seconds. |
