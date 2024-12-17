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

1. **Smart Scraping (Sync)**

```python
from pydantic import BaseModel


class ProductSchema(BaseModel):
    name: str
    price: float
    description: str


schema = [ProductSchema]
result = scrapegraph_tool.scrapegraph_smartscraper(
    prompt="Extract product information",
    url="https://example.com/product",
    api_key="your-api-key",
    schema=schema,
)
```

2. **Smart Scraping (Async)**

```python
result = await scrapegraph_tool.scrapegraph_smartscraper_async(
    prompt="Extract product information",
    url="https://example.com/product",
    api_key="your-api-key",
    schema=schema,
)
```

3. **Submit Feedback**

```python
response = scrapegraph_tool.scrapegraph_feedback(
    request_id="request-id",
    api_key="your-api-key",
    rating=5,
    feedback_text="Great results!",
)
```

4. **Check Credits**

```python
credits = scrapegraph_tool.scrapegraph_get_credits(api_key="your-api-key")
```

## Requirements

- Python 3.8+
- `scrapegraph-py` package
- Valid Scrapegraph API key

## License

This project is licensed under the MIT License - see the LICENSE file for details.
