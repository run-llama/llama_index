# LlamaIndex Tool: MCP Discovery

This tool allows LlamaIndex agents to query a Model Context Protocol (MCP) Discovery server to find and retrieve other tools dynamically.

## ⚠️ Important Note

This MCP Discovery integration **does not work out of the box**.
It requires a **separately deployed MCP Discovery server**, which you must **self-host locally or deploy to your own cloud**.

This tool acts only as a **client** and assumes an existing, reachable MCP Discovery server.

---

## Required Environment Variables

```env
SUPABASE_URL=your-supabase-url
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
OPENAI_API_KEY=your-openai-api-key
```

## Deploying the MCP Discovery Server

```bash
  git clone https://github.com/yksanjo/mcp-discovery.git
  cd mcp-discovery
  npm install
```

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

# Initialize the tool with the MCP Discovery API
tool_spec = MCPDiscoveryTool(
    api_url="https://mcp-discovery-two.vercel.app/api/v1/discover"
)

# Convert the spec to a list of FunctionTools
tools = tool_spec.to_tool_list()

# Create an agent with the discovery tool
agent = ReActAgent.from_tools(tools, verbose=True)

# The agent can now use the 'discover_tools' function to find MCP servers it needs
agent.chat("Find me a server that can send Slack notifications")
```

## API Response Format

This tool uses the standard MCP Discovery response schema as defined in the [MCP Discovery](https://github.com/yksanjo/mcp-discovery). The API should return responses following this format:

```json
{
  "recommendations": [
    {
      "server": "filesystem-server",
      "name": "Filesystem Server",
      "npm_package": "@modelcontextprotocol/server-filesystem",
      "install_command": "npx -y @modelcontextprotocol/server-filesystem",
      "confidence": 0.85,
      "description": "Secure file operations for MCP...",
      "category": "development",
      "github_url": "https://github.com/modelcontextprotocol/servers"
    }
  ],
  "total_found": 10,
  "query_time_ms": 52
}
```

**Note**: To optimize context window usage, the tool summarizes the raw JSON into a concise string containing only the name, server, and category. This allows the LLM to efficiently evaluate and select the best tool without being overwhelmed by installation metadata.

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
