# Firecrawl Web Loader

## Instructions for Firecrawl Web Loader

### Setup and Installation

1. **Install Firecrawl Package**: Ensure the `firecrawl-py` package is installed to use the Firecrawl Web Loader. Install it via pip with the following command:

```bash
pip install 'firecrawl-py>=4.3.3'
```

2. **API Key**: Secure an API key from [Firecrawl.dev](https://www.firecrawl.dev/) to access the Firecrawl services.

### Using Firecrawl Web Loader

- **Initialization**: Initialize the `FireCrawlWebReader` by providing the API key, the desired mode of operation (`crawl`, `scrape`, `map`, `search`, or `extract`), and any optional parameters for the Firecrawl API.

```python
from llama_index.readers.web.firecrawl_web.base import FireCrawlWebReader

firecrawl_reader = FireCrawlWebReader(
    api_key="your_api_key_here",
    mode="crawl",  # or "scrape" or "map" or "search" or "extract"
    # Common params for the underlying Firecrawl client
    # e.g. formats for content types and crawl limits
    params={
        "formats": ["markdown", "html"],  # for scrape or crawl
        "limit": 100,  # for crawl
        # "scrape_options": {"formats": ["markdown", "html"]},  # alternative shape for crawl
    },
)
```

- **Loading Data**: To load data, use the `load_data` method with the URL you wish to process.

```python
# For crawl or scrape mode
documents = firecrawl_reader.load_data(url="http://example.com")
# For search mode
documents = firecrawl_reader.load_data(query="search term")
```

### Example Usage

Here is an example demonstrating how to initialize the FireCrawlWebReader, load documents from a URL, and then create a summary index from those documents for querying.

```python
# Initialize the FireCrawlWebReader with your API key and desired mode
firecrawl_reader = FireCrawlWebReader(
    api_key="your_api_key_here",  # Replace with your actual API key
    mode="crawl",  # Choose between "crawl", "scrape", "map", "search" and "extract"
    params={
        # Provide formats for the content you want to retrieve
        "formats": ["markdown", "html"],
        # Limit the number of pages to crawl
        "limit": 50,
    },
)

# Load documents from Paul Graham's essay URL
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
