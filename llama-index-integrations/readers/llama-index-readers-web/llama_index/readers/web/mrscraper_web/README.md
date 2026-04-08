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

## Usage

### Fetch HTML

```python
from llama_index.readers.web import MrScraperWebReader

reader = MrScraperWebReader(api_token="YOUR_TOKEN", mode="fetch_html")
documents = reader.load_data(url="https://example.com", geo_code="US")
print(documents[0].text)
```

### AI Scraper (create + run)

```python
reader = MrScraperWebReader(api_token="YOUR_TOKEN", mode="scrape")
documents = reader.load_data(
    url="https://example.com/products",
    message="Extract all product names, prices, and ratings",
    agent="listing",
    proxy_country="US",
)
```

### Rerun AI Scraper

```python
reader = MrScraperWebReader(api_token="YOUR_TOKEN", mode="rerun_scraper")
documents = reader.load_data(
    scraper_id="scraper_12345",
    url="https://example.com/category/electronics",
)
```

### Bulk Rerun AI Scraper

```python
reader = MrScraperWebReader(
    api_token="YOUR_TOKEN", mode="bulk_rerun_ai_scraper"
)
documents = reader.load_data(
    scraper_id="scraper_12345",
    urls=["https://example.com/item1", "https://example.com/item2"],
)
```

### Standalone Methods

Each SDK function is also available as a standalone method:

```python
reader = MrScraperWebReader(api_token="YOUR_TOKEN")

# Sync
docs = reader.fetch_html("https://example.com")
docs = reader.create_scraper("https://example.com/products", "Extract prices")
docs = reader.rerun_scraper("scraper_123", "https://example.com/page2")
docs = reader.get_all_results(page_size=20, sort_order="DESC")
docs = reader.get_result_by_id("result_456")

# Async
docs = await reader.fetch_html("https://example.com")
docs = await reader.create_scraper(
    "https://example.com/products", "Extract prices"
)
```

## API Token

Get your API token at [https://app.mrscraper.com](https://app.mrscraper.com).
