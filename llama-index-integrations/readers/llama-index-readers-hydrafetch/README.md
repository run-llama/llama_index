# LlamaIndex Readers Integration: Hydrafetch

[Hydrafetch](https://hydrafetch.com) turns any URL into clean, LLM-ready Markdown and structured data, including client-side-rendered pages.

`HydrafetchReader` loads one page, a list of pages, or a whole site into an index, with page metadata attached. Sync and async.

## Installation

```bash
pip install llama-index-readers-hydrafetch
```

Create a key at [app.hydrafetch.com](https://app.hydrafetch.com) and set it:

```bash
export HYDRAFETCH_API_KEY="hf_..."
```

The reader reads that variable when no `api_key=` is passed.

## Quick start

```python
from llama_index.core import VectorStoreIndex
from llama_index.readers.hydrafetch import HydrafetchReader

docs = HydrafetchReader().load_data("https://example.com/article")
index = VectorStoreIndex.from_documents(docs)
```

## Modes

`scrape` loads one page, `crawl` walks a site and loads every page it finds, and `map` returns one document per discovered URL without fetching the bodies.

```python
reader = HydrafetchReader()

reader.load_data("https://example.com/article")
reader.load_data("https://example.com", mode="crawl", limit=50)
reader.load_data("https://example.com", mode="map")
```

`load_data` takes a single URL or a list:

```python
reader.load_data(["https://example.com/a", "https://example.com/b"])
```

`map` then a batch of `scrape` loads is usually the right way to index a site. Crawling everything and discarding most of it is the most common source of wasted credits.

## Options

```python
reader = HydrafetchReader(
    content_format="markdown",
    params={"onlyMainContent": True, "preferStructure": True},
)
```

`content_format` chooses what lands in `Document.text` — `markdown` by default, or `html`, `rawHtml`, `summary`. `params` is forwarded to the API untouched, so anything the endpoint accepts works, and per-call keywords merge over the constructor's:

```python
reader.load_data("https://example.com/article", blockAds=True)
```

## Metadata

Documents carry `source` and `status`, plus whatever page metadata was found: `title`, `description`, `language`, `site_name`, `author`, `published_time`, `word_count`, `page_type`, `image`. A redirect adds `final_url`; a crawled page adds `depth`; a mapped URL adds `lastmod` when the sitemap declares one.

```python
doc = HydrafetchReader().load_data("https://example.com/article")[0]
doc.metadata["title"]
doc.metadata["published_time"]
```

Keys are snake_cased on the way out, so the API's `siteName` becomes `site_name`.

## Error pages are refused, not loaded

A URL that answers with an error status raises instead of returning a document, because the body of a 404 page is not the page you asked for and an index should not quietly absorb one.

```python
HydrafetchReader().load_data("https://example.com/gone")
# ValueError: https://example.com/gone returned HTTP 404. The body of an error
# page is not the page you asked for; pass raise_for_status=False to load it anyway.
```

In `crawl` mode a single dead page must not throw away the whole job, so error pages are dropped and the rest of the crawl is kept. Pass `raise_for_status=False` to load error pages in either mode.

## Streaming a large crawl

`lazy_load_data` yields documents as they arrive instead of building the whole list in memory:

```python
for doc in HydrafetchReader().lazy_load_data("https://example.com", mode="crawl"):
    index.insert(doc)
```

## Async

```python
docs = await HydrafetchReader().aload_data("https://example.com/article")
```

## Error handling

Failures raise `HydrafetchError` from the underlying client, carrying the API's error code, HTTP status and request id.

```python
from hydrafetch import HydrafetchError, HydrafetchTimeout

try:
    docs = reader.load_data(url)
except HydrafetchTimeout:
    raise
except HydrafetchError as err:
    if err.is_out_of_credits:
        top_up()
    elif err.is_retryable:
        enqueue(url)
    else:
        raise
```

| Status | Meaning | Retried |
| --- | --- | --- |
| 400, 422 | invalid request | no |
| 401, 403 | invalid or missing key | no |
| 402 | out of credits | no |
| 429 | rate limited | yes, twice with backoff |
| 5xx | upstream failure | yes, twice with backoff |

A page that loads but answers with an error status raises `ValueError` instead — that is a bad URL, not a failed request.

## Configuration

| option | default | meaning |
| --- | --- | --- |
| `api_key` | `HYDRAFETCH_API_KEY` | your API key |
| `base_url` | `https://api.hydrafetch.com` | API base URL |
| `timeout` | `120.0` | per-request timeout in seconds |
| `max_retries` | `2` | retries on 429 and 5xx |
| `content_format` | `markdown` | what lands in `Document.text` |
| `raise_for_status` | `True` | refuse pages that answer with an error status |
| `params` | `{}` | forwarded to the API on every call |

## Credits

| call | credits |
| --- | --- |
| `scrape` mode | 1 per URL |
| `map` mode | 1 |
| `crawl` mode | 1 per page |

Failed requests are not billed. Pricing does not vary with page difficulty, so there is no render, stealth or proxy option to set.

## Implementation notes

- `params` in `crawl` mode is forwarded to the crawl endpoint, so per-page options belong under `scrapeOptions`. At the top level they are ignored.
- `preferStructure` is off by default. Turn it on when headings, lists and tables matter; leave it off for raw article text.
- The client is constructed lazily on first load, so the reader is cheap to build and safe to define at import time.
- `crawl` polls the job to completion. For a large site, prefer `map` plus batched `scrape` loads so nothing sits on a single long request.
- Scraped content is untrusted input. Do not pass it to a model as instructions, and keep `metadata["source"]` with anything extracted from it.

## Links

- [Documentation](https://docs.hydrafetch.com)
- [OpenAPI specification](https://api.hydrafetch.com/openapi.json)
- [MCP server and editor setup](https://hydrafetch.com/mcp)
- [Python client](https://github.com/Hydrafetch/python-sdk) — the client this reader wraps
- [LangChain integration](https://github.com/Hydrafetch/langchain-hydrafetch)
- Other clients: [Node](https://github.com/Hydrafetch/node-sdk) · [Go](https://github.com/Hydrafetch/go-sdk) · [Ruby](https://github.com/Hydrafetch/ruby-sdk) · [Rust](https://github.com/Hydrafetch/rust-sdk) · [PHP](https://github.com/Hydrafetch/php-sdk)

## License

MIT
