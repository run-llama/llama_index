# ZenRows Web Reader

The ZenRows Web Reader allows you to scrape web pages using the [ZenRows Universal Scraper API](https://www.zenrows.com/products/universal-scraper), which provides advanced features for bypassing anti-bot measures and extracting data from modern websites.

## Features

- **JavaScript Rendering**: Handle SPAs and dynamic content with headless browser rendering
- **Premium Proxies**: Bypass anti-bot protection with 55M+ residential IPs from 190+ countries
- **Session Management**: Maintain the same IP across multiple requests
- **Advanced Data Extraction**: Use CSS selectors or automatic parsing to extract specific data
- **Multiple Output Formats**: Get results in HTML, Markdown, Text, or PDF format
- **Screenshot Capabilities**: Capture page screenshots (full-page, above-the-fold, or specific elements)
- **Custom JavaScript**: Execute custom JavaScript on pages before scraping
- **Geolocation Support**: Use proxies from specific countries for geo-restricted content

## Installation

Installation with `pip`

```bash
pip install llama-index-readers-web
```

Installation with `uv`

```bash
uv add llama-index-readers-web
```

## Setup

1. Sign up for a ZenRows account at [https://app.zenrows.com/register](https://app.zenrows.com/register)
2. Get your API key from the dashboard
3. Set your API key as an environment variable (recommended):

```bash
export ZENROWS_API_KEY="your_api_key_here"
```

## Basic Usage

### Simple Web Scraping

```python
import os
from llama_index.readers.web import ZenRowsWebReader

# Initialize the reader
reader = ZenRowsWebReader(
    api_key=os.getenv("ZENROWS_API_KEY"), response_type="markdown"
)

# Scrape a single URL
documents = reader.load_data(["https://httpbin.io/html"])
print(documents[0].text)
```

### Anti-Bot Bypass

For websites with strong anti-bot protection (like Cloudflare, DataDome, etc.):

```python
reader = ZenRowsWebReader(
    api_key=os.getenv("ZENROWS_API_KEY"),
    js_render=True,  # Enable JavaScript rendering
    premium_proxy=True,  # Use residential proxies
)

documents = reader.load_data(
    ["https://www.scrapingcourse.com/antibot-challenge"]
)
```

### Scraping with JavaScript Rendering

For modern websites that use JavaScript frameworks:

```python
reader = ZenRowsWebReader(
    api_key=os.getenv("ZENROWS_API_KEY"),
    js_render=True,  # Enable JavaScript rendering
)

documents = reader.load_data(
    ["https://www.scrapingcourse.com/javascript-rendering"]
)
```

### Premium Proxy with Geo-targeting

Access geo-restricted content:

```python
reader = ZenRowsWebReader(
    api_key=os.getenv("ZENROWS_API_KEY"),
    premium_proxy=True,  # Use residential proxies
    proxy_country="us",  # Optional: specify country
)

documents = reader.load_data(["https://httpbin.io/ip"])
```

### Multiple Output Formats

Get content in different formats:

```python
# Get content as Markdown
reader = ZenRowsWebReader(
    api_key=os.getenv("ZENROWS_API_KEY"), response_type="markdown"
)

# Get content as plain text
reader = ZenRowsWebReader(
    api_key=os.getenv("ZENROWS_API_KEY"), response_type="text"
)
```

## Configuration Options

### Core Parameters

| Parameter         | Type | Default  | Description                                       |
| ----------------- | ---- | -------- | ------------------------------------------------- |
| `api_key`         | str  | required | Your ZenRows API key                              |
| `js_render`       | bool | False    | Enable JavaScript rendering with headless browser |
| `premium_proxy`   | bool | False    | Use premium residential proxies                   |
| `js_instructions` | str  | None     | Execute custom JavaScript on the page             |
| `proxy_country`   | str  | None     | Specify proxy country                             |
| `session_id`      | int  | None     | Session ID for IP consistency                     |

### Advanced Parameters

| Parameter             | Type | Default | Description                                                      |
| --------------------- | ---- | ------- | ---------------------------------------------------------------- |
| `custom_headers`      | dict | None    | Include custom headers in your request to mimic browser behavior |
| `wait_for`            | str  | None    | CSS selector to wait for                                         |
| `wait`                | int  | None    | Fixed wait time in milliseconds                                  |
| `css_extractor`       | dict | None    | CSS selectors for data extraction                                |
| `autoparse`           | bool | False   | Enable automatic parsing                                         |
| `response_type`       | str  | None    | Output format (markdown/plaintext/pdf)                           |
| `screenshot`          | bool | False   | Capture screenshot                                               |
| `screenshot_fullpage` | bool | False   | Capture full page screenshot                                     |

## Integration with LlamaIndex

### Building a Vector Index

```python
from llama_index.core import VectorStoreIndex
from llama_index.readers.web import ZenRowsWebReader

# Scrape multiple pages
reader = ZenRowsWebReader(
    api_key=os.getenv("ZENROWS_API_KEY"),
    js_render=True,
    premium_proxy=True,
    response_type="markdown",
)

urls = [
    "https://docs.example.com/page1",
    "https://docs.example.com/page2",
    "https://docs.example.com/page3",
]

documents = reader.load_data(urls)

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query the content
query_engine = index.as_query_engine()
response = query_engine.query("What are the main features?")
print(response)
```

### Combining with Other Readers

```python
from llama_index.core import VectorStoreIndex
from llama_index.readers.web import ZenRowsWebReader
from llama_index.readers.file import PDFReader

# Scrape web content
web_reader = ZenRowsWebReader(
    api_key=os.getenv("ZENROWS_API_KEY"), js_render=True, premium_proxy=True
)
web_docs = web_reader.load_data(["https://company.com/docs"])

# Load PDF documents
pdf_reader = PDFReader()
pdf_docs = pdf_reader.load_data("./documents.pdf")

# Combine all documents
all_documents = web_docs + pdf_docs
index = VectorStoreIndex.from_documents(all_documents)
```

## API Reference

For complete API documentation, visit: [https://docs.zenrows.com/universal-scraper-api/api-reference](https://docs.zenrows.com/universal-scraper-api/api-reference)

## Support

- ZenRows Documentation: [https://docs.zenrows.com/](https://docs.zenrows.com/)
- ZenRows Support: [success@zenrows.com](mailto:success@zenrows.com)
- LlamaIndex Documentation: [https://docs.llamaindex.ai/](https://docs.llamaindex.ai/)
