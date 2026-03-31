# Querit Search Tool

This tool connects to the [Querit Search API](https://querit.ai) and allows an Agent to perform web searches with optional filters for language, geography, site whitelist/blacklist, and time range.

## Installation

```bash
pip install llama-index-tools-querit
```

## Usage

```python
from llama_index.tools.querit import QueritToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = QueritToolSpec(api_key="YOUR_QUERIT_API_KEY")

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(await agent.run("Search for the latest news about AI in 2025"))
print(await agent.run("Search for Python tutorials from the United States only"))
```

## Available Tools

- `search`: Basic web search by query text.
- `search_with_language`: Search filtered by language (e.g. `"english"`, `"japanese"`, `"korean"`, `"german"`, `"french"`, `"spanish"`, `"portuguese"`).
- `search_with_geo`: Search restricted to a specific country (e.g. `"united states"`, `"japan"`, `"united kingdom"`).
- `search_with_site_filter`: Search with domain whitelist and/or blacklist.
- `search_with_time_range`: Search filtered to a time range (e.g. `"d1"` past day, `"w1"` past week, `"m1"` past month, `"y1"` past year).

This loader is designed to be used as a way to load data as a Tool in an Agent.
