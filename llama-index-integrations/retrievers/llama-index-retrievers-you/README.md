# LlamaIndex Retrievers Integration: You Retriever

Retriever for You.com's Search API, providing unified web and news search results.

## Installation

```bash
pip install llama-index-retrievers-you
```

## Usage

```python
from llama_index.retrievers.you import YouRetriever

# Initialize with API key
retriever = YouRetriever(api_key="your-api-key")

# Or set YDC_API_KEY environment variable
# retriever = YouRetriever()

# Retrieve search results synchronously
results = retriever.retrieve("your search query")

# Retrieve search results asynchronously
results = await retriever.aretrieve("your search query")
```

## Features

- Unified web and news search
- Customizable search parameters (country, language, freshness, etc.)
- Optional livecrawl for full-page content in html or markdown format
- Seamless integration with LlamaIndex query engines and agents

## API Reference

See the [You.com API documentation](https://docs.you.com/api-reference/search/v1-search) for details on available parameters.

## Development

### Setup

```bash
# Install package with dev dependencies
pip install -e ".[dev]"

# Or using uv
uv pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```
