# Outcrawl Web Loader

This loader reads web pages through the [Outcrawl](https://outcrawl.ai) API. It has two modes: `scrape` returns one URL as Markdown, and `search` runs a web search and returns one document per result.

## Setup

```bash
pip install llama-index-readers-web
```

You need an Outcrawl API key. Pass it as `api_key` or set `OUTCRAWL_API_KEY`.

## Usage

### Scrape mode

```python
from llama_index.readers.web import OutcrawlWebReader

reader = OutcrawlWebReader(mode="scrape")  # reads OUTCRAWL_API_KEY
documents = reader.load_data(url="https://en.wikipedia.org/wiki/Web_crawler")

# Extra request fields are passed through, e.g. the whole page instead of the main content
documents = reader.load_data(
    url="https://example.com", params={"onlyMainContent": False}
)
```

Each document's text is the page Markdown; its metadata holds `source`, `status_code`, `title`, `description` and `language` when present.

### Search mode

```python
from llama_index.readers.web import OutcrawlWebReader

reader = OutcrawlWebReader(mode="search", params={"limit": 5})
documents = reader.load_data(query="llamaindex web readers")

# Render every result and use its Markdown as the document text
documents = reader.load_data(
    query="llamaindex web readers", params={"scrape": True}
)
```

Each document's metadata holds `source`, `title`, `snippet`, `rank` and `query`.

### With an index

```python
from llama_index.core import SummaryIndex

index = SummaryIndex.from_documents(documents)
response = index.as_query_engine().query("What do these pages say?")
```

API reference: https://outcrawl.ai/docs/api/overview
