# fastCRW Web Loader

## Instructions for fastCRW Web Loader

[fastCRW](https://fastcrw.com/) is an open-source (AGPL-3.0) web scraper that ships as a single ~8MB binary. Run it against the managed cloud, point it at your own server, or run the engine locally with no server at all.

### Setup and Installation

1. **Install crw Package**: Ensure the `crw` package is installed to use the fastCRW Web Loader. Install it via pip with the following command:

```bash
pip install crw
```

2. **API Key**: Secure an API key from [fastCRW](https://fastcrw.com/dashboard) to access the managed cloud, and set it as the `CRW_API_KEY` environment variable (or pass it directly). A self-hosted server may require no auth at all, and the local engine (`CRW_LOCAL=1`) needs no key.

### Using fastCRW Web Loader

- **Initialization**: Initialize the `CrwWebReader` by providing the API key (or relying on `CRW_API_KEY`), the desired mode of operation (`crawl`, `scrape`, `map`, or `search`), and any optional parameters for the fastCRW API. To target a self-hosted deployment, pass `api_url`.

```python
from llama_index.readers.web.crw_web.base import CrwWebReader

crw_reader = CrwWebReader(
    api_key="your_api_key_here",  # or set CRW_API_KEY in the environment
    mode="crawl",  # or "scrape" or "map" or "search"
    # api_url="http://localhost:3000",  # optional: self-hosted deployment
    # Common params for the underlying crw client
    # e.g. formats for content types and crawl limits
    params={
        "formats": ["markdown", "html"],  # for scrape or crawl
        "maxPages": 100,  # for crawl
    },
)
```

- **Running the engine locally**: set `CRW_LOCAL=1` in the environment and the client runs the fastCRW engine as a local subprocess, with no server and no API key. Anti-bot handling, JS rendering and proxy rotation are part of the open-source engine, so the local path is not a reduced one.

```python
import os

os.environ["CRW_LOCAL"] = "1"

crw_reader = CrwWebReader(mode="scrape", params={"formats": ["markdown"]})
```

- **Loading Data**: To load data, use the `load_data` method with the URL you wish to process.

```python
# For crawl or scrape mode
documents = crw_reader.load_data(url="http://example.com")
# For search mode
documents = crw_reader.load_data(query="search term")
```

### Example Usage

Here is an example demonstrating how to initialize the CrwWebReader, load documents from a URL, and then create a summary index from those documents for querying.

```python
# Initialize the CrwWebReader with your API key and desired mode
crw_reader = CrwWebReader(
    api_key="your_api_key_here",  # Replace with your actual API key
    mode="crawl",  # Choose between "crawl", "scrape", "map" and "search"
    params={
        # Provide formats for the content you want to retrieve
        "formats": ["markdown", "html"],
        # Limit the number of pages to crawl
        "maxPages": 50,
    },
)

# Load documents from Paul Graham's essay URL
documents = crw_reader.load_data(url="http://www.paulgraham.com/")

# Create a summary index from the loaded documents for querying
index = SummaryIndex.from_documents(documents)

# Convert the summary index into a query engine
query_engine = index.as_query_engine()

# Perform a query on the index to find insights from Paul Graham's essays
response = query_engine.query("Insights from Paul Graham's essays")

# Display the query response
print(response)
```
