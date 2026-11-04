# LlamaIndex Tools Integration: Serpex

Search the web from a LlamaIndex agent with [Serpex](https://serpex.dev), a real-time web search API for AI agents. Each search returns ranked web results (title, URL and snippet), optionally with the top pages' content as markdown, as LlamaIndex `Document` objects.

## Installation

```bash
pip install llama-index-tools-serpex
```

## Usage

```python
import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.serpex import SerpexToolSpec

serpex_tool = SerpexToolSpec(api_key="your_serpex_api_key")

agent = FunctionAgent(
    tools=serpex_tool.to_tool_list(), llm=OpenAI(model="gpt-4.1")
)
print(asyncio.run(agent.run("What are the latest AI developments?")))
```

The agent sees one tool, `search(query, num_results)`.

### Search directly

```python
results = serpex_tool.search("recent AI news after:2026-09-01", num_results=5)

for doc in results:
    print(doc.metadata["title"], doc.metadata["url"])
```

Search operators go inside the query: `site:example.com`, `-site:example.com`,
`after:YYYY-MM-DD` and `before:YYYY-MM-DD`. `num_results` (1 to 10) limits how many
results are returned; the API returns at most 10.

### Page content

```python
serpex_tool = SerpexToolSpec(include_content=True, content_results=5)  # or 10
docs = serpex_tool.search("what is retrieval augmented generation")
```

The first `content_results` documents carry the page markdown after the snippet. A page that
could not be fetched keeps only the snippet and has `metadata["content_error"]` (for example
`"robots_disallowed"`). Credits: 1 per search; with page content, 1 when no page came back, 3 for
1-5 pages, 6 for 6-10 pages.

### Document metadata

`title`, `url`, `snippet`, `position`, `number_of_results`, `response_time`, `credits_used`,
`query`, plus `content_error` when a page could not be fetched. `engine` is deprecated and always
`"auto"`.

### Errors

A failed request raises `SerpexAPIError`, a subclass of `requests.exceptions.HTTPError`, whose
message says what happened, whether it was charged and whether to retry:

| Status | Example message                                                                                             |
| ------ | ----------------------------------------------------------------------------------------------------------- |
| 400    | `Serpex rejected the request (HTTP 400): content_results must be 5 or 10.`                                  |
| 401    | `Serpex rejected the API key (HTTP 401): Invalid or inactive API key. Check SERPEX_API_KEY.`                |
| 402    | `Out of Serpex credits (HTTP 402). Balance: 0 credits. Top up at https://app.serpex.dev/dashboard/billing.` |
| 429    | `Serpex rate limit exceeded (limit: 5 requests per second). Retry in 292 ms.`                               |
| 5xx    | `Serpex is temporarily unavailable (HTTP 503): ... Not charged.`                                            |

`SerpexAPIError` also carries `status_code`, `details` (the API's error body) and `retryable`.
Rate limits are 5, 25 or 50 requests per second depending on the plan.

### Deprecated arguments

`engine` (constructor and `search()`) and `time_range` (`search()`) are deprecated and ignored
by the Serpex API. They are still accepted so existing code keeps working, emit a
`DeprecationWarning` and are not sent. For a date filter, put `after:YYYY-MM-DD` /
`before:YYYY-MM-DD` in the query.

## API Key

Get your API key from the [Serpex dashboard](https://app.serpex.dev/dashboard).

Set it as an environment variable:

```bash
export SERPEX_API_KEY=your_api_key
```

## Links

- [Serpex website](https://serpex.dev)
- [Serpex documentation](https://serpex.dev/docs)
- [LlamaIndex](https://llamaindex.ai)
