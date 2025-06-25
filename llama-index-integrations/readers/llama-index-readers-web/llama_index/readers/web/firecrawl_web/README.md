# Firecrawl Web Loader

## Instructions for Firecrawl Web Loader

### Setup and Installation

1. **Install Firecrawl Package**: Ensure the `firecrawl-py` package is installed to use the Firecrawl Web Loader. Install it via pip with the following command:

```bash
pip install firecrawl-py
```

2. **API Key**: Secure an API key from [Firecrawl.dev](https://www.firecrawl.dev/) to access the Firecrawl services.

### Using Firecrawl Web Loader

The FireCrawlWebReader supports four modes of operation:

- **scrape**: Extract content from a single URL
- **crawl**: Crawl an entire website and extract content from all accessible pages
- **search**: Search for content across the web
- **extract**: Extract structured data from URLs using a prompt

#### Initialization

Initialize the FireCrawlWebReader by providing the API key, the desired mode, and parameters:

```python
from llama_index.readers.web.firecrawl_web.base import FireCrawlWebReader

# Scrape a single page
scrape_reader = FireCrawlWebReader(
    api_key="your_api_key_here",
    mode="scrape",
    params={"timeout": 30}
)

# Crawl a website
crawl_reader = FireCrawlWebReader(
    api_key="your_api_key_here",
    mode="crawl",
    params={"max_depth": 2, "limit": 10}
)

# Search for content
search_reader = FireCrawlWebReader(
    api_key="your_api_key_here",
    mode="search",
    params={"limit": 5}
)

# Extract structured data
extract_reader = FireCrawlWebReader(
    api_key="your_api_key_here",
    mode="extract",
    params={"prompt": "Extract the main topics"}
)
```

#### Loading Data

Load data using the `load_data` method with the appropriate parameters for each mode:

```python
# For scrape or crawl mode
documents = reader.load_data(url="http://example.com")

# For search mode
documents = reader.load_data(query="search term")

# For extract mode
documents = reader.load_data(urls=["http://example1.com", "http://example2.com"])
```

### Example Usage

Here is a complete example demonstrating how to use the FireCrawlWebReader with different modes:

```python
from llama_index.readers.web.firecrawl_web.base import FireCrawlWebReader
from llama_index.core import SummaryIndex

# Initialize the FireCrawlWebReader for crawling
firecrawl_reader = FireCrawlWebReader(
    api_key="your_api_key_here",  # Replace with your actual API key
    mode="crawl",
    params={
        "max_depth": 2,
        "limit": 10,
        "delay": 1
    }
)

# Load documents from Paul Graham's website
documents = firecrawl_reader.load_data(url="http://www.paulgraham.com/")

# Create a summary index from the loaded documents for querying
index = SummaryIndex.from_documents(documents)

# Convert the summary index into a query engine
query_engine = index.as_query_engine()

# Perform a query on the index to find insights from Paul Graham's essays
response = query_engine.query("Insights from Paul Graham's essays")

# Display the query response
print(response)
```

### Available Parameters

The `params` parameter accepts different options depending on the mode. Check the documentation for each of the methods:

- [Scrape](https://docs.firecrawl.dev/api-reference/endpoint/scrape)
- [Crawl](https://docs.firecrawl.dev/api-reference/endpoint/crawl-post)
- [Search](https://docs.firecrawl.dev/api-reference/endpoint/search)
- [Extract](https://docs.firecrawl.dev/api-reference/endpoint/extract)
