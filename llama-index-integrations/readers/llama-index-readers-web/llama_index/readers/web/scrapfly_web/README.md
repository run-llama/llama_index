# ScrapFly Web Loader

## Instructions for ScrapFly Web Loader

### Setup and Installation

1. **Install ScrapFly Python SDK**: Install `scrapfly-sdk` Python package is installed to use the ScrapFly Web Loader. Install it via pip with the following command:

   ```bash
   pip install scrapfly-sdk llama-index-readers-web
   ```

2. **API Key**: Register for free from [scrapfly.io/register](https://www.scrapfly.io/register/) to obtain your API key.

### Using ScrapFly Web Loader

- **Initialization**: Initialize the ScrapflyReader by providing the API key and the optional scrape config to use.

  ```python
  from llama_index.readers.web import ScrapflyReader

  # Initiate ScrapflyReader with your ScrapFly API key
  scrapfly_reader = ScrapflyReader(
      api_key="Your ScrapFly API key",  # Get your API key from https://www.scrapfly.io/
      ignore_scrape_failures=True,  # Ignore unprocessable web pages and log their exceptions
  )

  scrapfly_scrape_config = {
      "asp": True,  # Bypass scraping blocking and antibot solutions, like Cloudflare
      "render_js": True,  # Enable JavaScript rendering with a cloud headless browser
      "proxy_pool": "public_residential_pool",  # Select a proxy pool (datacenter or residnetial)
      "country": "us",  # Select a proxy location
      "auto_scroll": True,  # Auto scroll the page
      "js": "",  # Execute custom JavaScript code by the headless browser
  }

  # Load documents from URLs as markdown
  documents = scrapfly_reader.load_data(
      urls=["https://web-scraping.dev/products"],
      scrape_config=scrapfly_scrape_config,  # Pass the scrape config
      scrape_format="markdown",  # The scrape result format, either `markdown`(default) or `text`
  )
  ```

See the [ScrapFly documentation](https://scrapfly.io/docs/scrape-api/getting-started) for the full details on using the ScrapeConfig.

### Example Usage

Here is an example demonstrating how to initialize the ScrapflyReader, load documents from a URL, and then create a summary index from those documents for querying.

```python
from llama_index.core import SummaryIndex
from llama_index.readers.web import ScrapflyReader

# Initiate ScrapflyReader with your ScrapFly API key
scrapfly_reader = ScrapflyReader(
    api_key="Your ScrapFly API key",  # Get your API key from https://www.scrapfly.io/
    ignore_scrape_failures=True,  # Ignore unprocessable web pages and log their exceptions
)

# Load product data document as markdown
documents = scrapfly_reader.load_data(
    urls=["https://web-scraping.dev/products"]
)
# Create a summary index from the loaded documents for querying
index = SummaryIndex.from_documents(documents)

# Convert the summary index into a query engine
query_engine = index.as_query_engine()

# Perform a query on the index to find insights from scraped products
response = query_engine.query("What is the Dark Energy Potion flavor?")

# Display the query response
print(response)
"Bold cherry cola flavor"
```
