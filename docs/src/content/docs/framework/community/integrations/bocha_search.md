---
title: Bocha Web Search
---

[Bocha Web Search](https://bochaai.com/) is a web search API that returns structured
results enriched with metadata such as title, URL, site name, and publication date.

The `llama-index-tools-bocha-search` package is an **external** LlamaIndex
[ToolSpec](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/)
maintained at
[github.com/NiTingKY/llama-index-tools-bocha-search](https://github.com/NiTingKY/llama-index-tools-bocha-search).
It follows LlamaIndex's external-integration guidance and is published independently
on PyPI.

## Installation

```bash
pip install llama-index-tools-bocha-search
```

## What It Provides

| Detail | Value |
|---|---|
| Package name | `llama-index-tools-bocha-search` |
| Import path | `llama_index.tools.bocha_search` |
| ToolSpec class | `BochaSearchToolSpec` |
| Tool exposed to agents | `web_search` |
| Auth | `api_key` parameter or `BOCHA_SEARCH_API_KEY` env var |
| Endpoint override | `api_url` parameter or `BOCHA_SEARCH_API_URL` env var |
| Result type | LlamaIndex `Document` objects with `title`, `url`, `site_name`, and `published_date` metadata |

## Usage

```python
import os
from llama_index.tools.bocha_search import BochaSearchToolSpec

bocha_spec = BochaSearchToolSpec(
    api_key=os.getenv("BOCHA_SEARCH_API_KEY"),
    default_count=5,
)

tools = bocha_spec.to_tool_list()
```

### Using with an Agent

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.bocha_search import BochaSearchToolSpec

bocha_spec = BochaSearchToolSpec(api_key=os.getenv("BOCHA_SEARCH_API_KEY"))

agent = FunctionAgent(
    tools=bocha_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4o-mini"),
)

response = await agent.run("What are the highlights of the latest LlamaIndex release?")
print(str(response))
```

## Example Notebook

A full walkthrough — including direct tool usage and agent integration — is available
in the [Bocha Web Search example notebook](/python/examples/tools/bocha_web_search).

## References

- [Bocha Web Search](https://bochaai.com/)
- [llama-index-tools-bocha-search on GitHub](https://github.com/NiTingKY/llama-index-tools-bocha-search)
- [llama-index-tools-bocha-search on PyPI](https://pypi.org/project/llama-index-tools-bocha-search/)
