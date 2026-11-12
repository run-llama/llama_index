# A2A Tools for LlamaIndex

[![PyPI version](https://badge.fury.io/py/llama-index-tools-a2a.svg)](https://badge.fury.io/py/llama-index-tools-a2a)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Agent-to-Agent (A2A) Protocol integration for LlamaIndex**

Enable your LlamaIndex agents to discover and invoke other A2A-compatible agents as tools. This follows Google's A2A protocol specification for universal agent interoperability.

## Features

- **Outbound Client**: Connect to any A2A agent via its AgentCard URL
- **Skill Discovery**: Automatically discover available tools/skills from remote agents
- **Schema Conversion**: Maps A2A skill schemas to LlamaIndex FunctionTools
- **Async Support**: Full async/await support for non-blocking operation
- **Partial Params**: Support for pre-filling common parameters across tools

## Installation

```bash
pip install llama-index-tools-a2a
```

Or with uv:
```bash
uv add llama-index-tools-a2a
```

## Quick Start

### Discovering Tools from an A2A Agent

```python
from llama_index.tools.a2a import aget_tools_from_a2a_card

# Connect to a remote A2A agent and get its tools
tools = await aget_tools_from_a2a_card(
    "https://weather-agent.example.com/.well-known/agent-card.json"
)

# Use tools in your LlamaIndex workflow
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools(tools)
result = await agent.chat("What's the weather in Tokyo?")
```

### Filtering Specific Tools

```python
# Only use specific skills
tools = await aget_tools_from_a2a_card(
    "https://agent.example.com/.well-known/agent-card.json",
    allowed_tools=["search_web", "calculate"]
)
```

### Pre-filling Common Parameters

```python
# Apply default parameters across all tools
tools = await aget_tools_from_a2a_card(
    "https://agent.example.com/.well-known/agent-card.json",
    global_partial_params={"api_key": "my-secret-key"}
)
```

### Exposing a Workflow as an A2A Agent (Inbound)

```python
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step
from llama_index.tools.a2a import workflow_as_a2a

class MyStartEvent(StartEvent):
    query: str

class MyWorkflow(Workflow):
    @step
    async def run(self, start_event: MyStartEvent) -> StopEvent:
        return StopEvent(result=f"Processed: {start_event.query}")

# Create the A2A-compatible FastAPI app
app = workflow_as_a2a(MyWorkflow())

# Run with uvicorn: uvicorn my_module:app --host 0.0.0.0 --port 8000
# The agent will be discoverable at http://localhost:8000/.well-known/agent-card.json
```

## Architecture

This integration mirrors the existing `llama-index-tools-mcp` pattern:

| Concept | MCP Equivalent | A2A Implementation |
|---------|---------------|-------------------|
| Tool discovery | `list_tools()` | AgentCard.skills[] |
| Tool invocation | `call_tool(name, args)` | `message/send` + `tasks/get` |
| Inbound server | `MCPServer` | ASGI app (`workflow_as_a2a()`) |
| Workflow wrapping | `workflow_as_mcp()` | `workflow_as_a2a()` |

## Project Status

**Milestone 1**: Package scaffolded ✅
**Milestone 2**: Outbound client with AgentCard discovery ✅
**Milestone 3**: Inbound ASGI server (`workflow_as_a2a()`) ✅
**Milestone 4**: Integration tests ✅
**Milestone 5**: Documentation & cookbook ✅

## Cookbook / Examples

See [`examples/a2a.ipynb`](examples/a2a.ipynb) for a full end-to-end Jupyter notebook demonstrating:
- Creating an inbound A2A agent with `workflow_as_a2a()`
- Connecting as an outbound client via `BasicA2AClient`
- Converting skills to LlamaIndex FunctionTools with `A2AToolSpec`
- Using A2A tools inside a LlamaIndex ReActAgent
- Filtering skills and pre-filling partial parameters

Run the example server standalone:
```bash
python examples/a2a_server.py --port 8000
# Then visit http://localhost:8000/.well-known/agent-card.json
```

## Contributing

Contributions are welcome! Please see the [LlamaIndex contributing guide](https://github.com/run-llama/llama_index/blob/main/CONTRIBUTING.md) for details.

This contribution is being made in collaboration with the Google A2A team under issue [#24104](https://github.com/run-llama/llama_index/issues/24104).

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Google A2A Team for protocol design and mentorship
- LlamaIndex community for the MCP integration pattern this builds upon
