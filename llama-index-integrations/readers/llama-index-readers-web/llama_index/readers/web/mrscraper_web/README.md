# MrScraper Web Reader

`MrScraperWebReader` integrates [MrScraper](https://mrscraper.com) with LlamaIndex, providing AI-powered web scraping and data extraction capabilities.

## Installation

```bash
pip install llama-index-readers-web mrscraper-sdk
```

## Features

| Mode                        | Description                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------- |
| `fetch_html`                | Fetch rendered HTML via a stealth browser with JS rendering and bot-detection evasion |
| `scrape`                    | Create an AI scraper with natural-language instructions                               |
| `rerun_scraper`             | Rerun an existing AI scraper on a new URL                                             |
| `bulk_rerun_ai_scraper`     | Batch-rerun an AI scraper on multiple URLs                                            |
| `rerun_manual_scraper`      | Rerun a dashboard-created manual scraper on a new URL                                 |
| `bulk_rerun_manual_scraper` | Batch-rerun a manual scraper on multiple URLs                                         |
| `get_all_results`           | Retrieve paginated scraping results                                                   |
| `get_result_by_id`          | Retrieve a single result by ID                                                        |

## AI agents (`scrape` mode)

When `mode="scrape"`, you choose an **`agent`** that matches how the page is structured. All agents share the same optional parameters in this reader; the API applies them according to agent type.

| Agent         | Typical use                                                           | Parameters to know                                                                                                                                                                                                                                                            |
| ------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`general`** | Single-page or focused extraction from one URL                        | **`proxy_country`** (optional): proxy geolocation. Crawl-style knobs below are mainly relevant for **`map`**.                                                                                                                                                                 |
| **`listing`** | Catalog / search / listing pages (many similar rows, often paginated) | **`max_pages`** (default `50`): cap how many listing pages to follow. Also supports **`proxy_country`**.                                                                                                                                                                      |
| **`map`**     | Crawling multiple linked pages on a site                              | **`max_depth`** (default `2`): link depth. **`max_pages`** (default `50`): max pages to visit. **`limit`** (default `1000`): max records collected. **`include_patterns`** / **`exclude_patterns`**: optional URL pattern filters (strings as expected by the MrScraper API). |

Defaults if you omit optional arguments: `max_depth=2`, `max_pages=50`, `limit=1000`, `include_patterns=""`, `exclude_patterns=""`.

## Usage

### Fetch HTML

```python
from llama_index.readers.web import MrScraperWebReader

reader = MrScraperWebReader(api_token="YOUR_TOKEN", mode="fetch_html")
documents = reader.load_data(
    url="https://example.com",
    geo_code="US",  # optional; default "US"
    timeout=120,  # optional; seconds
    block_resources=False,  # optional; block images/CSS/fonts
)
print(documents[0].text)
```

### AI scraper — create and run (`scrape`)

**Required:** `url`, `message` (natural-language extraction instructions).

**Optional:** `agent`, `proxy_country`, `max_depth`, `max_pages`, `limit`, `include_patterns`, `exclude_patterns` (see [AI agents](#ai-agents-scrape-mode) above).

```python
# General: one-page extraction
reader = MrScraperWebReader(api_token="YOUR_TOKEN", mode="scrape")
documents = reader.load_data(
    url="https://example.com/product/123",
    message="Extract title, price, and bullet points as JSON.",
    agent="general",
    proxy_country="US",  # optional
)

# Listing: catalog-style pages; constrain listing pagination with max_pages
documents = reader.load_data(
    url="https://example.com/search?q=shoes",
    message="Extract each product: name, price, url.",
    agent="listing",
    max_pages=3,  # optional; listing pagination cap
    proxy_country="US",
)

# Map: crawl the site with depth, page cap, record limit, URL include/exclude
documents = reader.load_data(
    url="https://example.com/blog",
    message="Extract article title, date, and body for each post.",
    agent="map",
    max_depth=2,  # optional
    max_pages=50,  # optional
    limit=500,  # optional
    include_patterns="*/blog/*",  # optional; API-specific pattern string
    exclude_patterns="*/tag/*",  # optional
)
```

`Document.metadata` includes `scraper_id` when the API returns it (useful for `rerun_scraper`).

### Rerun AI scraper (`rerun_scraper`)

Runs an **existing** AI scraper (by id) against a **new** URL. The scraper’s agent and behavior were fixed when it was created; this call only overrides **crawl limits and URL filters** for this run.

**Required:** `scraper_id`, `url`.

**Optional:** `max_depth`, `max_pages`, `limit`, `include_patterns`, `exclude_patterns` (same defaults as in code: `2`, `50`, `1000`, empty strings).

```python
reader = MrScraperWebReader(api_token="YOUR_TOKEN", mode="rerun_scraper")
documents = reader.load_data(
    scraper_id="scraper_12345",
    url="https://example.com/another-category",
    max_depth=2,
    max_pages=30,
    limit=200,
    include_patterns="",
    exclude_patterns="*/admin/*",
)
```

There is no `agent` argument on rerun; the scraper id already identifies the configured agent.

### Bulk rerun AI scraper

```python
reader = MrScraperWebReader(
    api_token="YOUR_TOKEN", mode="bulk_rerun_ai_scraper"
)
documents = reader.load_data(
    scraper_id="scraper_12345",
    urls=["https://example.com/item1", "https://example.com/item2"],
)
```

### Manual scraper rerun modes

`rerun_manual_scraper` and `bulk_rerun_manual_scraper` only take `scraper_id` and `url` / `urls` (no AI agent or crawl knobs in this reader).

### All results — paginated (`get_all_results`)

**Optional:** `sort_field`, `sort_order`, `page_size`, `page`, `search`, `date_range_column`, `start_at`, `end_at`.

```python
reader = MrScraperWebReader(api_token="YOUR_TOKEN", mode="get_all_results")
documents = reader.load_data(
    sort_field="updatedAt",
    sort_order="DESC",
    page_size=20,
    page=1,
    # Optional filters
    search="product",
    date_range_column="createdAt",  # e.g. "createdAt" or "updatedAt"
    start_at="2026-03-01",  # format: YYYY-MM-DD
    end_at="2026-04-01",
)
```

When the API returns a non-empty list of result rows, you get **one `Document` per row**; each `Document.text` is JSON for that item. If there are no rows, you get a single `Document` whose `text` is the stringified full API payload. Example:

```python
import json

for doc in documents:
    row = json.loads(doc.text)
    # Safer access
    result_id = row.get("id")
    status = row.get("status")
```

### Result by ID (`get_result_by_id`)

```python
reader = MrScraperWebReader(api_token="YOUR_TOKEN", mode="get_result_by_id")
documents = reader.load_data(result_id="result_456")
```

## Standalone methods (SDK-aligned)

These mirror the MrScraper client helpers and are **async** — use `await` inside `async def`, or run them with `asyncio.run(...)`.

```python
import asyncio
from llama_index.readers.web import MrScraperWebReader

reader = MrScraperWebReader(api_token="YOUR_TOKEN")


async def main():
    docs = await reader.fetch_html("https://example.com")
    docs = await reader.create_scraper(
        "https://example.com/products",
        "Extract product name and price",
        agent="listing",
        max_pages=10,
    )
    docs = await reader.rerun_scraper(
        "scraper_123",
        "https://example.com/page2",
        max_pages=20,
    )
    docs = await reader.get_all_results(page_size=20, sort_order="DESC")
    docs = await reader.get_result_by_id("result_456")
    return docs


asyncio.run(main())
```

## API Token

Get your API token at [https://app.mrscraper.com](https://app.mrscraper.com).
