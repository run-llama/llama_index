# LlamaIndex Tool: Ejentum Reasoning Harness

Wraps the hosted [Ejentum](https://ejentum.com) MCP server as a LlamaIndex tool spec. Exposes four cognitive-harness tools (`harness_reasoning`, `harness_code`, `harness_anti_deception`, `harness_memory`) an agent calls before generating, each returning a structured scaffold the agent absorbs to harden its next response against named failure modes.

## Installation

```bash
pip install llama-index-tools-ejentum
```

## Requirements

- Python >= 3.10
- `llama-index-core` >= 0.13.0,<0.15 (transitive)
- `llama-index-tools-mcp` >= 0.4.0 (transitive)
- `EJENTUM_API_KEY` environment variable. Free and paid tiers at <https://ejentum.com/auth/register>.

## Usage

### Minimal

```python
import os
from llama_index.tools.ejentum import EjentumToolSpec

os.environ["EJENTUM_API_KEY"] = "..."  # or set in your shell

spec = EjentumToolSpec()
tools = spec.to_tool_list()
```

### Subset of modes

```python
# Only expose reasoning and code harnesses
spec = EjentumToolSpec(modes=["reasoning", "code"])
tools = spec.to_tool_list()
```

Valid mode names: `reasoning`, `code`, `anti_deception`, `memory`.

### With a ReActAgent

```python
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.ejentum import EjentumToolSpec

tools = EjentumToolSpec().to_tool_list()
agent = ReActAgent.from_tools(tools, llm=OpenAI(model="gpt-4o-mini"))

response = await agent.achat(
    "Why might our microservice return 503s only under specific load patterns?"
)
```

The agent routes to `harness_reasoning` based on the tool description. The returned scaffold (failure pattern, procedure, suppression vectors, falsification test) is fed back into the agent's context for the next generation step.

## What each harness returns

Each `harness_*` call returns four labeled sections:

- `[NEGATIVE GATE]` / `[DECEPTION PATTERN]` / `[CODE FAILURE]` / `[PERCEPTION FAILURE]`: the named failure mode to avoid.
- `[PROCEDURE]` (or `[INTEGRITY PROCEDURE]`): an executable step-by-step the model follows internally.
- `Amplify:` and `Suppress:` vectors: signals to activate or block.
- `[FALSIFICATION TEST]` (or `[INTEGRITY CHECK]`): verification criterion to self-check the next response.

Section labels differ per harness (see the [Ejentum docs](https://ejentum.com/docs) for the per-mode field map).

## About the underlying MCP server

This package is a thin subclass of [`llama-index-tools-mcp`](https://pypi.org/project/llama-index-tools-mcp/)'s `McpToolSpec`, pre-configured with the hosted endpoint and Bearer authentication. For raw MCP usage against arbitrary servers, use `McpToolSpec` directly.

The same MCP server is available across other surfaces: stdio via `npx -y ejentum-mcp`, hosted at `https://api.ejentum.com/mcp` (Streamable HTTP), and listed on the [Official MCP Registry](https://registry.modelcontextprotocol.io/) as `io.github.ejentum/ejentum-mcp`.

## Links

- [Source repository](https://github.com/ejentum/ejentum-mcp)
- [Install guide for other clients](https://ejentum.com/docs/mcp_guide)
- [MCP Registry entry](https://registry.modelcontextprotocol.io/v0/servers?search=io.github.ejentum/ejentum-mcp)
