# LlamaIndex Agent Module Tool

A LlamaIndex tool spec for querying **Agent Module** — deterministic EU AI Act compliance knowledge built for autonomous agents.

Returns binary logic gates and specific statutory citations. No probabilistic inference — all records have `confidence_required: 1.0`.

## Installation

```bash
pip install llama-index-tools-agent-module
```

## Quick Start

```python
from llama_index.tools.agent_module import AgentModuleToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

# Initialize with your API key
tool_spec = AgentModuleToolSpec(am_key="YOUR_AM_KEY")
tools = tool_spec.to_tool_list()

agent = ReActAgent.from_tools(tools, llm=OpenAI(model="gpt-4o"), verbose=True)

# Query compliance knowledge
response = agent.chat(
    "Does our emotion recognition system in hiring qualify as high-risk AI "
    "under the EU AI Act, and if so, what FRIA obligations apply?"
)
```

## Getting an API Key

**Free trial:** `https://agent-module.dev/waitlist`
Or call the MCP tool `get_trial_key` if you have the Agent Module MCP server configured.

**Production keys:** `https://agent-module.dev` (Inaugural cohort: $19/month)

## Available Tools

| Tool | Module | Coverage |
|---|---|---|
| `query_module` | Any ETH_XXX | Direct module lookup by ID |
| `query_fria` | ETH_021 | FRIA obligations — Art. 27, Aug 2026 deadline |
| `query_prohibited_practices` | ETH_016 | Prohibited AI — Art. 5 (€35M penalty tier) |
| `query_high_risk_classification` | ETH_015 | High-risk classification — Art. 6, Annex III |
| `query_risk_management` | ETH_017 | Risk management system — Art. 9 |
| `query_conformity_assessment` | ETH_013 | Conformity assessment — Art. 43 |
| `query_gpai_obligations` | ETH_020 | GPAI model obligations — Art. 53-55 |

## Usage Without an Agent

```python
from llama_index.tools.agent_module import AgentModuleToolSpec

spec = AgentModuleToolSpec(am_key="YOUR_AM_KEY")

# Check if your system requires a FRIA
fria_data = spec.query_fria()

# Check prohibited practices
prohibited = spec.query_prohibited_practices()

# Direct lookup by module ID
result = spec.query_module("ETH_015")
```

## MCP Alternative

Agent Module also exposes an MCP server. For LlamaIndex + MCP:

```python
# Configure via MCP server at https://api.agent-module.dev/mcp
# Transport: streamable_http | Auth: X-AM-Key header
```

Full module list: `https://agent-module.dev/core-ethics.md`