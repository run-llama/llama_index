# LlamaIndex Tools Integration: Octen

This tool connects to [Octen](https://octen.ai), a fast real-time web search API for AI, to enable your agent to search and retrieve content from the Internet.

To begin, you need to obtain an API key at [octen.ai](https://octen.ai).

## Installation

```bash
pip install llama-index-tools-octen
```

## Usage

```python
import os
from llama_index.tools.octen import OctenToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

octen_tool = OctenToolSpec(
    api_key=os.environ["OCTEN_API_KEY"],
)
agent = FunctionAgent(
    tools=octen_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

response = await agent.run(
    "What are the latest developments in AI?"
)
print(response)
```

### Basic Search

```python
from llama_index.tools.octen import OctenToolSpec

tool = OctenToolSpec(api_key=os.environ["OCTEN_API_KEY"])

results = tool.search("latest AI research papers", num_results=5)
for r in results:
    print(r["title"], r["url"])
```

### Search with Domain Filtering

```python
results = tool.search(
    "machine learning",
    include_domains=["arxiv.org", "papers.nips.cc"],
)

results = tool.search(
    "Python tutorial",
    exclude_domains=["youtube.com", "reddit.com"],
)
```

### Search with Time Filter

```python
tool = OctenToolSpec(api_key=os.environ["OCTEN_API_KEY"])

results = tool.search(
    "Python release notes",
    start_time="2025-01-01",
    end_time="2026-03-31",
    time_basis="published",
)
```

### Retrieve Full Page Content

```python
tool = OctenToolSpec(api_key=os.environ["OCTEN_API_KEY"], max_characters=3000)

# Returns List[Document], each Document.text contains the full page content
docs = tool.search_and_retrieve_documents(
    query="Python 3.13 new features",
    num_results=3,
    include_domains=["docs.python.org"],
)

for doc in docs:
    print(doc.metadata["title"], doc.metadata["url"])
    print(doc.text[:200])
```

### Retrieve Highlighted Snippets

```python
# Returns List[Document], each Document.text contains a concise snippet
highlights = tool.search_and_retrieve_highlights(
    query="LlamaIndex agent tutorial",
    num_results=5,
    highlight_max_tokens=300,
)

for doc in highlights:
    print(doc.metadata["title"])
    print(doc.text)
```

### Text Filters and Safe Search

```python
results = tool.search(
    "Python web framework",
    include_text=["async"],
    exclude_text=["deprecated"],
    safesearch="strict",
    format="markdown",
)
```

## Available Tools

- `search`: Search the web using Octen for a list of results matching a natural language query.

- `search_and_retrieve_documents`: Search and retrieve full page content as LlamaIndex Documents.

- `search_and_retrieve_highlights`: Search and retrieve highlighted snippets as LlamaIndex Documents.

- `current_date`: Utility for the agent to get today's date (useful for time-filtered searches).

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | required | Octen API key |
| `verbose` | `bool` | `True` | Print search metadata |
| `max_results` | `int` | `5` | Default number of results |
| `max_characters` | `int` | `2000` | Max characters for full content retrieval |
| `timeout` | `float` | `None` | Request timeout in seconds |

### Per-query Parameters

All search functions support:

| Parameter | Type | Description |
|---|---|---|
| `query` | `str` | Natural language search query |
| `num_results` | `int` | Number of results (overrides default) |
| `include_domains` | `list[str]` | Restrict to these domains |
| `exclude_domains` | `list[str]` | Exclude these domains |
| `search_type` | `str` | `auto`, `keyword`, or `semantic` |
| `start_time` | `str` | Start date filter (ISO 8601) |
| `end_time` | `str` | End date filter (ISO 8601) |
| `time_basis` | `str` | `auto`, `published`, or `crawled` |
| `include_text` | `list[str]` | Text that must appear in results |
| `exclude_text` | `list[str]` | Text to exclude from results |
| `safesearch` | `str` | `off` or `strict` |
| `format` | `str` | `text` or `markdown` |

`search_and_retrieve_highlights` additionally supports:

| Parameter | Type | Description |
|---|---|---|
| `highlight_max_tokens` | `int` | Max tokens for highlighted snippets (default: 200) |
