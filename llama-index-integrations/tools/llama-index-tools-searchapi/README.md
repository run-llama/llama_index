# LlamaIndex Tools Integration: SearchApi.io

**`llama-index-tools-searchapi`** is a [LlamaIndex](https://www.llamaindex.ai/) tool
integration that gives agents real-time web-search capabilities powered by
[SearchApi.io](https://www.searchapi.io/).

SearchApi.io is a real-time SERP API that scrapes and structures results from
Google Search, Google News, Google Scholar, Google Images, and 50+ other
engines.  The `SearchApiToolSpec` wraps four of these engines as first-class
LlamaIndex tools, each returning a list of `Document` objects that slot
directly into any downstream LlamaIndex pipeline.

---

## Installation

```bash
pip install llama-index-tools-searchapi
```

> **Prerequisites** — You will also need an active SearchApi.io account.  A
> free tier is available at <https://www.searchapi.io/>.

---

## Quick Start

```python
import os
from llama_index.tools.searchapi import SearchApiToolSpec

# The API key can be passed directly or read from the environment.
os.environ["SEARCHAPI_API_KEY"] = "your-api-key-here"

tool_spec = SearchApiToolSpec()

# Web search
docs = tool_spec.search("LlamaIndex RAG pipelines", num=5)
for doc in docs:
    print(doc.metadata["title"], "—", doc.metadata["link"])
    print(doc.text[:200])
    print()
```

---

## Available Tools

| Tool | Engine | Description |
|---|---|---|
| `search` | `google` | Standard web search returning organic results |
| `news_search` | `google_news` | Google News results with source and date metadata |
| `scholar_search` | `google_scholar` | Academic papers with publication info and citation count |
| `image_search` | `google_images` | Image results with thumbnail and original-URL metadata |

### Common Parameters

All four methods share the same signature:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | — | Search query string |
| `num` | `int` | `10` | Max results to return |
| `gl` | `str \| None` | `None` | Geo-target country (ISO 3166-1 alpha-2, e.g. `"us"`) |
| `hl` | `str \| None` | `None` | Language code (e.g. `"en"`) |
| `location` | `str \| None` | `None` | Free-text location (e.g. `"London, UK"`) |

---

## Returned Document Structure

Each method returns `List[Document]`.  The `text` field contains the result
snippet (or image title); the `extra_info` dict holds engine-specific metadata:

### `search` / `news_search`

```python
doc.text          # snippet text
doc.metadata = {
    "title":    "Result title",
    "link":     "https://...",
    "position": 1,          # search / None for news
    # news only:
    "source":   "BBC News",
    "date":     "2 hours ago",
}
```

### `scholar_search`

```python
doc.text          # abstract snippet
doc.metadata = {
    "title":            "Paper title",
    "link":             "https://...",
    "publication_info": "Smith et al., Nature 2023",
    "cited_by_count":   412,
}
```

### `image_search`

```python
doc.text          # image title
doc.metadata = {
    "link":     "https://page-containing-image/",
    "original": "https://direct-url-to-full-size-image.jpg",
    "source":   "example.com",
}
```

---

## Usage with a LlamaIndex Agent

```python
import os
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.searchapi import SearchApiToolSpec

os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["SEARCHAPI_API_KEY"] = "your-searchapi-key"

tool_spec = SearchApiToolSpec()
tools = tool_spec.to_tool_list()

agent = OpenAIAgent.from_tools(
    tools,
    llm=OpenAI(model="gpt-4o"),
    verbose=True,
)

response = agent.chat(
    "What are the top research papers on retrieval-augmented generation "
    "from 2024, and how many citations do they have?"
)
print(response)
```

---

## Authentication

The API key is resolved in this order:

1. The `api_key` constructor argument.
2. The `SEARCHAPI_API_KEY` environment variable.

```python
# Option 1 — constructor arg
tool_spec = SearchApiToolSpec(api_key="sa-xxxx")

# Option 2 — environment variable
import os
os.environ["SEARCHAPI_API_KEY"] = "sa-xxxx"
tool_spec = SearchApiToolSpec()
```

---

## Examples

See the [`examples/`](examples/) directory for Jupyter notebooks.

---

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
make test

# Format & lint
make format
make lint
```

---

## License

MIT — see [LICENSE](LICENSE).
