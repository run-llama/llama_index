# LlamaIndex Tools - PraisonAI

This tool connects LlamaIndex agents to a [PraisonAI](https://github.com/MervinPraison/PraisonAI) server for running multi-agent workflows.

**Documentation**: https://docs.praison.ai/docs/

## Installation

```bash
pip install llama-index-tools-praisonai
```

## Usage

```python
from llama_index.tools.praisonai import PraisonAIToolSpec

# Initialize the tool
tool_spec = PraisonAIToolSpec(api_url="http://localhost:8080")

# Get tools for use with an agent
tools = tool_spec.to_tool_list()

# Or use directly
result = tool_spec.run_workflow("Research AI trends")
print(result)
```

## Available Methods

- `run_workflow(query)`: Run a query through the multi-agent workflow
- `run_agent(query, agent)`: Run a query through a specific agent
- `list_agents()`: List available agents

## Prerequisites

Start a PraisonAI server:

```bash
pip install praisonai
praisonai serve agents.yaml --port 8080
```
