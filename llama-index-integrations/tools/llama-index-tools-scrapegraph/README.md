# LlamaIndex Tool - Scrapegraph

This tool integrates [Scrapegraph](https://scrapegraphai.com) with LlamaIndex, providing intelligent web scraping capabilities with structured data extraction.

## Installation

```bash
pip install llama-index-tools-scrapegraph
```

## Usage

First, import and initialize the ScrapegraphToolSpec:

```python
from llama_index.tools.scrapegraph import ScrapegraphToolSpec

scrapegraph_tool = ScrapegraphToolSpec()
```

### Available Functions

The tool provides the following capabilities:

1. **Smart Scraper**

```python
from pydantic import BaseModel


# Define your schema (optional)
class ProductSchema(BaseModel):
    name: str
    price: float
    description: str


schema = [ProductSchema]

# Perform the scraping
result = scrapegraph_tool.scrapegraph_smartscraper(
    prompt="Extract product information",
    url="https://example.com/product",
    api_key="your-api-key",
    schema=schema,  # Optional
)
```

2. **Markdownify**

Convert webpage content to markdown format:

```python
markdown_content = scrapegraph_tool.scrapegraph_markdownify(
    url="https://example.com", api_key="your-api-key"
)
```

3. **Local Scrape**

Extract structured data from raw text:

```python
text = """
Your raw text content here...
"""

structured_data = scrapegraph_tool.scrapegraph_local_scrape(
    text=text, api_key="your-api-key"
)
```

## Requirements

- Python 3.8+
- `scrapegraph-py` package
- Valid Scrapegraph API key
