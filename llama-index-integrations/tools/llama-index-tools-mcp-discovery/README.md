# LlamaIndex Tool: MCP Discovery

This tool allows LlamaIndex agents to query a Model Context Protocol (MCP) Discovery server to find and retrieve other tools dynamically.

## Features

- 🔍 **Autonomous Tool Discovery**: Query MCP servers to discover available tools based on natural language descriptions
- ⚡ **Async Operations**: Built with `aiohttp` for high-performance async operations
- 🤖 **Seamless Integration**: Works directly with LlamaIndex agents via `BaseToolSpec`
- 🛡️ **Error Handling**: Graceful error handling with informative messages

## Requirements

- Python >= 3.9
- llama-index-core >= 0.13.0
- aiohttp >= 3.8.0

## Installation

```bash
pip install llama-index-tools-mcp-discovery
```

## Usage

```python
from llama_index.tools.mcp_discovery import MCPDiscoveryTool
from llama_index.core.agent import ReActAgent

# Initialize the tool with the discovery server URL
tool_spec = MCPDiscoveryTool(api_url="http://localhost:8000/api")

# Convert the spec to a list of FunctionTools
tools = tool_spec.to_tool_list()

# Create an agent with the discovery tool
agent = ReActAgent.from_tools(tools, verbose=True)

# The agent can now use the 'discover_tools' function to find what it needs
agent.chat("Find me a tool that can help with math")
```

## API Response Format

The MCP discovery API should return responses in the following format:

```json
{
  "recommendations": [
    {
      "name": "math-calculator",
      "description": "A tool for performing mathematical calculations",
      "category": "math"
    },
    {
      "name": "equation-solver",
      "description": "Solves algebraic equations",
      "category": "math"
    }
  ],
  "total_found": 2
}
```

## Examples

See the [`examples`](./examples) directory for more usage examples.

## Development

Run tests:

```bash
make test
```

Run linters:

```bash
make lint
```

Format code:

```bash
make format
```
