# Code Execution MCP Server

A Model Context Protocol (MCP) server that exposes battle-tested code execution capabilities.

This MCP server allows any AI agent (Claude, Cursor, Windsurf, etc.) to execute terminal commands and Python code on the host system with persistent sessions.

## Features

- **Execute Terminal Commands**: Run shell commands with full session persistence
- **Execute Python Code**: Run Python code via IPython with session management
- **Multiple Sessions**: Maintain separate execution contexts
- **Smart Output Handling**: Automatic prompt detection, timeout management, and dialog detection
- **Cross-Platform**: Works on Linux and macOS (Windows experimental via WSL)

## MCP Client Configuration (no installation needed)

Add to your application MCP config:

### Simple case using uvx

```json
{
  "mcpServers": {
    "code-execution": {
      "command": "uvx",
      "args": ["llama-index-tools-mcp-code-execution"]
    }
  }
}
```

### Or using pipx

```json
{
  "mcpServers": {
    "code-execution": {
      "command": "pipx",
      "args": ["run", "llama-index-tools-mcp-code-execution"]
    }
  }
}
```

## Configuration

The MCP server can be configured via environment variables:

| Variable | Default | Description |
|---|---|---|
| `CODE_EXEC_EXECUTABLE` | `/bin/bash` | Shell executable path |
| `CODE_EXEC_INIT_COMMANDS` | *(empty)* | Semicolon-separated init commands |
| `CODE_EXEC_FIRST_OUTPUT_TIMEOUT` | `30` | Wait for first output (seconds) |
| `CODE_EXEC_BETWEEN_OUTPUT_TIMEOUT` | `15` | Wait between output chunks (seconds) |
| `CODE_EXEC_DIALOG_TIMEOUT` | `5` | Detect dialog prompts (seconds) |
| `CODE_EXEC_MAX_EXEC_TIMEOUT` | `180` | Maximum execution time (seconds) |
| `CODE_EXEC_LOG_DIR` | *(empty)* | Log directory (empty = disabled) |

### Example: Custom shell with virtual environment

```json
{
  "mcpServers": {
    "code-execution": {
      "command": "uvx",
      "args": ["llama-index-tools-mcp-code-execution"],
      "env": {
        "CODE_EXEC_EXECUTABLE": "/bin/zsh",
        "CODE_EXEC_INIT_COMMANDS": "source /path/to/venv/bin/activate",
        "CODE_EXEC_LOG_DIR": "/path/to/logs"
      }
    }
  }
}
```

### Example: Override timeouts

```json
{
  "mcpServers": {
    "code-execution": {
      "command": "uvx",
      "args": ["llama-index-tools-mcp-code-execution"],
      "env": {
        "CODE_EXEC_FIRST_OUTPUT_TIMEOUT": "60",
        "CODE_EXEC_MAX_EXEC_TIMEOUT": "300"
      }
    }
  }
}
```

## Manual Installation

```bash
cd llama-index-integrations/tools/llama-index-tools-mcp-code-execution
pip install -e .
```

Then configure your MCP client:

```json
{
  "mcpServers": {
    "code-execution": {
      "command": "code-execution-mcp"
    }
  }
}
```

## Usage with LlamaIndex Agents (Anthropic Claude)

Connect this MCP server to a LlamaIndex agent powered by Anthropic's Claude:

```bash
pip install llama-index-tools-mcp llama-index-tools-mcp-code-execution llama-index-llms-anthropic
export ANTHROPIC_API_KEY="your-api-key"
```

```python
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.anthropic import Anthropic
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

async def main():
    # Connect to the code-execution MCP server via stdio
    mcp_client = BasicMCPClient("code-execution-mcp", timeout=30)

    # Convert MCP tools to LlamaIndex FunctionTool objects
    tool_spec = McpToolSpec(client=mcp_client)
    tools = await tool_spec.to_tool_list_async()

    # Create an Anthropic-powered agent
    agent = FunctionAgent(
        name="CodeAgent",
        description="An agent that can execute code.",
        llm=Anthropic(model="claude-sonnet-4-20250514", max_tokens=4096),
        tools=tools,
        system_prompt="You are a coding assistant with terminal and Python execution tools.",
    )

    response = await agent.run("What Python version is installed? Run python3 --version.")
    print(response)

asyncio.run(main())
```

See [examples/](examples/) for more complete usage examples.

## Available Tools

### `execute_terminal`

Execute a terminal command in the specified session.

**Parameters:**
- `command` (string, required): The shell command to execute
- `session` (integer, optional): Session number (default: 0)

**Returns:** The accumulated terminal output from the session.

### `execute_python`

Execute Python code via IPython in the specified session.

**Parameters:**
- `code` (string, required): The Python code to execute
- `session` (integer, optional): Session number (default: 0)

**Returns:** The accumulated IPython output from the session.

### `get_output`

Get accumulated output from a terminal session.

**Parameters:**
- `session` (integer, optional): Session number (default: 0)

**Returns:** The accumulated terminal output from the session.

### `reset_terminal`

Reset a terminal session, closing and reopening it.

**Parameters:**
- `session` (integer, optional): Session number (default: 0)
- `reason` (string, optional): Reason for the reset

**Returns:** Text confirmation for the agent.

## Session Management

- Sessions (terminal instances) allow maintaining separate execution contexts
- Each session can be used and reset individually
- Sessions persist until reset
- Session 0 is the default
- Any session number can be used

## Virtual Environment Considerations

When the MCP server is launched from a virtual environment, shell sessions may NOT automatically inherit the venv activation.

**Solution:** Use init commands to explicitly activate your virtual environment:

```json
{
  "env": {
    "CODE_EXEC_INIT_COMMANDS": "source /path/to/venv/bin/activate"
  }
}
```

## Architecture

The server consists of these modules:

- `main.py` - FastMCP server with tool definitions
- `tty_session.py` - TTY session management with smart output handling
- `shell_local.py` - Local shell process spawning via pexpect
- `config.py` - Environment variable configuration
- `strings.py` - String manipulation (ANSI cleaning, prompt/dialog detection)
- `print_style.py` - Logging utilities

## Platform Support

- **Linux**: Fully supported
- **macOS**: Fully supported
- **Windows**: Experimental (use WSL for best results)

## Security

> **WARNING**: This MCP server allows full code execution on the host system. Security is the responsibility of the MCP client.

Only use with trusted AI agents and in controlled environments.

## License

MIT License - see [LICENSE](LICENSE) for details.
