# Haunt Tool

This tool connects LlamaIndex agents to [Haunt](https://hauntapi.com?utm_source=llamaindex&utm_medium=integration&utm_campaign=sweep-2026-07), a web extraction API built for agents.

Give it a URL and a plain-language prompt ("the product name, price and stock status") and get structured JSON back, no selectors. When a page cannot be read (bot wall, login wall, missing page), Haunt returns an honest `error_code` such as `access_denied`, `login_required`, or `not_found` instead of invented content, so the agent can branch on the failure. Failed reads are not charged.

## Installation

```bash
pip install llama-index-tools-haunt
```

Get a free API key (1,000 credits, no card) at [hauntapi.com](https://hauntapi.com/#signup) and set it as `HAUNT_API_KEY` or pass it to the spec.

## Usage

```python
from llama_index.tools.haunt import HauntToolSpec

haunt_tool = HauntToolSpec(api_key="your-key")

# Structured extraction for an agent
haunt_tool.extract(
    "https://news.ycombinator.com",
    "the top 5 story titles and their points",
)
# '{"stories": [{"title": "...", "points": 572}, ...]}'

# Load pages as markdown Documents for indexing
docs = haunt_tool.load(["https://example.com/docs/page-one"])
```

Within an agent:

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

agent = FunctionAgent(
    tools=HauntToolSpec(api_key="your-key").to_tool_list(),
    llm=OpenAI(model="gpt-4.1-mini"),
)
print(await agent.run("What is the top story on Hacker News right now?"))
```

An unreadable page in `load` raises with the honest reason; `extract` returns the error code as JSON for the agent to reason over.
