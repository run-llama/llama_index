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
    print(doc.text[:500])


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

```python
tool = MrScraperToolSpec(api_key="MRSCRAPER_API_TOKEN")

result = await tool.create_scraper(
    "https://example.com/products",
    "Extract all product names, prices, and ratings",
    agent="listing",
    proxy_country="US",
)
scraper_id = result["data"]["id"]
print("Scraper ID:", scraper_id)
```

### Rerun a Scraper on a New URL

```python
result = await tool.rerun_scraper(
    scraper_id=scraper_id,
    url="https://example.com/products?page=2",
)
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

```python
# All results (paginated)
page = await tool.get_all_results(
    sort_field="updatedAt",
    sort_order="DESC",
    page_size=20,
    page=1,
)
print(page["data"])

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
