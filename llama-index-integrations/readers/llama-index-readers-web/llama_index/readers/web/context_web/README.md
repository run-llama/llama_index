# LlamaIndex Context.dev Web Reader

`ContextWebReader` loads live web data from [Context.dev](https://www.context.dev)
as LlamaIndex `Document` objects. It supports page scraping, website crawling,
sitemap discovery, web search, and structured extraction through one reader.

## Installation

```bash
pip install llama-index-readers-web
```

Create an API key in the [Context.dev dashboard](https://context.dev/dashboard) and
set it in your environment:

```bash
export CONTEXT_DEV_API_KEY="ctxt_secret_..."
```

## Scrape a page

```python
from llama_index.readers.web import ContextWebReader

reader = ContextWebReader(mode="scrape")
documents = reader.load_data(url="https://www.context.dev")
```

## Crawl a website

```python
reader = ContextWebReader(
    mode="crawl",
    params={"maxPages": 25, "maxDepth": 2},
)
documents = reader.load_data(url="https://docs.context.dev")
```

## Discover sitemap URLs

```python
reader = ContextWebReader(
    mode="sitemap",
    params={"maxLinks": 100, "search": "authentication guides"},
)
documents = reader.load_data(url="context.dev")
```

## Search the web

```python
reader = ContextWebReader(
    mode="search",
    params={"markdownOptions": {"enabled": True}},
)
documents = reader.load_data(query="latest AI infrastructure launches")
```

## Extract structured data

```python
reader = ContextWebReader(
    mode="extract",
    params={
        "schema": {
            "type": "object",
            "properties": {"plans": {"type": "array"}},
            "required": ["plans"],
        },
        "instructions": "Extract every pricing plan.",
    },
)
documents = reader.load_data(url="https://www.context.dev/pricing")
```

Each mode also supports `aload_data` for asynchronous applications. See the
[Context.dev API documentation](https://docs.context.dev) for all endpoint
parameters.
