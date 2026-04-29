# Perplexity Search Tool

Search the web for up-to-date information using the [Perplexity Search API](https://docs.perplexity.ai/docs/search/quickstart). Each result is returned as a `Document` containing a snippet plus title, URL, and (when available) a publication date.

To use the tool you need a Perplexity API key — create one at https://www.perplexity.ai/account/api/keys.

## Installation

```bash
pip install llama-index-tools-perplexity-search
```

## Usage

```python
from llama_index.tools.perplexity_search import PerplexitySearchToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

# Reads PERPLEXITY_API_KEY (or PPLX_API_KEY) from the environment if
# api_key is not passed explicitly.
perplexity_tool = PerplexitySearchToolSpec(api_key="your-key")

agent = FunctionAgent(
    tools=perplexity_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
)

await agent.run("What were the biggest AI announcements last week?")
```

Direct invocation:

```python
docs = perplexity_tool.perplexity_search(
    "site-reliability engineering best practices",
    max_results=5,
    search_recency_filter="month",
    search_domain_filter=["-pinterest.com"],  # denylist; do not mix with allowlist
)
for doc in docs:
    print(doc.metadata["title"], doc.metadata["url"])
    print(doc.text[:200])
```

### Filters

- `search_domain_filter`: list of domains. Use bare domains to allowlist (`"nytimes.com"`) or prefix `-` to denylist (`"-pinterest.com"`). The API treats the list as either an allowlist or a denylist — never mix the two. See the [domain filter docs](https://docs.perplexity.ai/docs/search/filters/domain-filter).
- `search_recency_filter`: one of `"hour"`, `"day"`, `"week"`, `"month"`, `"year"`. See the [date/recency filter docs](https://docs.perplexity.ai/docs/search/filters/date-time-filters).

## Available Functions

`perplexity_search`: Run a search query and return a list of `Document` objects, one per result. `text` is the snippet; `metadata` contains `title`, `url`, and `date`.
