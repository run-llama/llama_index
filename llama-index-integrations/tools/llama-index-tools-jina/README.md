# LlamaIndex Tools Integration: Jina

## Installation

Ensure your system has Python installed and proceed with the following installations via pip:

```bash
pip install llama-index-tools-jina
```

## Jina Search Tool

This tool enables agents to search and retrieve results from the Jina search api. This tool will search the web and return the top five results with their URLs and contents, each in clean, LLM-friendly text.

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](./examples/jina_search.ipynb)

Here's an example usage of the JinaSearchToolSpec.

```python
from llama_index.tools.jina import JinaSearchToolSpec
from llama_index.agent.openai import OpenAIAgent

tool_spec = JinaSearchToolSpec()

agent = OpenAIAgent.from_tools(JinaSearchToolSpec.to_tool_list())

agent.chat("What's going on with the superconductor lk-99")
agent.chat("what are the latest developments in machine learning")
```

## Available tool functions:

- `jina_search`: Make a query to Jina API to receive an instant answer.
