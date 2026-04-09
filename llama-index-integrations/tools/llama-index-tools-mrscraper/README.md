# MrScraper Tool

This tool connects to the [MrScraper](https://mrscraper.com) web-scraping API and gives LlamaIndex agents the ability to:

- **Fetch rendered HTML** from any URL using a stealth browser (JS rendering, bot-detection evasion, geo-proxying).
- **Create AI-powered scrapers** that extract structured data from web pages using natural-language instructions.
- **Rerun scrapers** (AI or manual) on new URLs, including bulk/batch operations.
- **Retrieve scraping results** with pagination, sorting, and filtering.

## Installation

```bash
pip install llama-index-tools-mrscraper
```

## Authentication

You need a MrScraper API token. Get yours at <https://app.mrscraper.com>.

## Usage

### Basic — Fetch HTML

```python
import asyncio
from llama_index.tools.mrscraper import MrScraperToolSpec


async def main():
    tool = MrScraperToolSpec(api_key="MRSCRAPER_API_TOKEN")
    doc = await tool.fetch_html("https://example.com")
    print(doc)


asyncio.run(main())
```

### With a LlamaIndex Agent

```python
from llama_index.tools.mrscraper import MrScraperToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

mrscraper_tool = MrScraperToolSpec(api_key="MRSCRAPER_API_TOKEN")

agent = FunctionAgent(
    tools=mrscraper_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
)

response = await agent.run(
    "Scrape https://example.com/products and extract all product names and prices"
)
print(response)
```

### Create an AI Scraper

#### Listing Agent

Best for pages with repeated items (product listings, job boards, search results).

```python
tool = MrScraperToolSpec(api_key="MRSCRAPER_API_TOKEN")

result = await tool.create_scraper(
    "https://example.com/products",
    "Extract all product names, prices, and ratings",
    agent="listing",
    proxy_country="US",
    # Optional filters
    max_pages=2,  # Number
)
scraper_id = result["data"]["id"]
print("Scraper ID:", scraper_id)
```

#### Map Agent

Best for crawling all sub-pages or sitemaps of a site. Accepts additional parameters
to control crawl depth, page limits, and URL filtering patterns.

```python
tool = MrScraperToolSpec(api_key="MRSCRAPER_API_TOKEN")

result = await tool.create_scraper(
    "https://example.com",
    "Crawl the site and extract every blog post title and publish date",
    agent="map",
    proxy_country="US",
    # Optional filters
    max_depth=3,
    max_pages=100,
    limit=500,
    # URL filtering (IMPORTANT: use proper regex)
    include_patterns=r"^https:\/\/example\.com\/blog\/.*$",
    exclude_patterns=r"^https:\/\/example\.com\/blog\/archive\/.*$|^https:\/\/example\.com\/blog\/tags\/.*$",
)
scraper_id = result["data"]["id"]
print("Map Scraper ID:", scraper_id)
```

| Parameter          | Default | Description                                               |
| ------------------ | ------- | --------------------------------------------------------- |
| `max_depth`        | `2`     | How many link-levels deep the crawler should follow       |
| `max_pages`        | `50`    | Maximum number of pages to visit                          |
| `limit`            | `1000`  | Maximum number of records to extract                      |
| `include_patterns` | `""`    | `\|\|`-separated URL patterns the crawler **must** match  |
| `exclude_patterns` | `""`    | `\|\|`-separated URL patterns the crawler should **skip** |

### Rerun a Scraper on a New URL

#### Rerun a Listing Scraper

For listing scrapers, only `scraper_id` and `url` are required.

```python
result = await tool.rerun_scraper(
    scraper_id=scraper_id,
    url="https://example.com/products?page=2",
    # Optional filters
    max_pages=2,
)
print(result["data"])
```

#### Rerun a Map Scraper

For map scrapers you can also override crawl parameters per rerun.

```python
result = await tool.rerun_scraper(
    scraper_id=scraper_id,
    url="https://example.com/blog",
    # Optional filters
    max_depth=2,
    max_pages=75,
    limit=300,
    include_patterns="/blog/2025/*",
    exclude_patterns="/blog/2025/draft/*",
)
print(result["data"])
```

### Bulk Rerun on Multiple URLs

```python
result = await tool.bulk_rerun_ai_scraper(
    scraper_id=scraper_id,
    urls=[
        "https://example.com/products/item1",
        "https://example.com/products/item2",
        "https://example.com/products/item3",
    ],
)
```

### Retrieve Results

### Retrieve All Results by Range

```python
# All results (paginated)
page = await tool.get_all_results(
    sort_field="updatedAt",
    sort_order="DESC",
    page_size=20,
    page=1,
    # Optional filters
    search="product",
    date_range_column="createdAt",  # "createdAt" or "updatedAt"
    start_at="2026-03-01",  # format: YYYY-MM-DD
    end_at="2026-04-01",  # format: YYYY-MM-DD
)
data = page.get("data", [])
print(data)
```

```python
# A specific result by ID
result = await tool.get_result_by_id("result_12345")
print(result["data"])
```

## Available Functions

| Function                    | Description                                                           |
| --------------------------- | --------------------------------------------------------------------- |
| `fetch_html`                | Fetch rendered HTML via the MrScraper stealth browser                 |
| `create_scraper`            | Create & run an AI-powered scraper with natural-language instructions |
| `rerun_scraper`             | Rerun an AI scraper on a new URL                                      |
| `bulk_rerun_ai_scraper`     | Rerun an AI scraper on multiple URLs in one batch                     |
| `rerun_manual_scraper`      | Rerun a manually configured scraper on a single URL                   |
| `bulk_rerun_manual_scraper` | Rerun a manual scraper on multiple URLs in one batch                  |
| `get_all_results`           | List all results with filtering & pagination                          |
| `get_result_by_id`          | Fetch a single result by its ID                                       |

## AI Scraper Agent Types

| Agent       | Best used for                                |
| ----------- | -------------------------------------------- |
| `"general"` | Default; handles almost any page             |
| `"listing"` | Product listings, job boards, search results |
| `"map"`     | Crawling all sub-pages / sitemaps of a site  |

This tool is designed to be used as a way to load data as a Tool in an Agent.
