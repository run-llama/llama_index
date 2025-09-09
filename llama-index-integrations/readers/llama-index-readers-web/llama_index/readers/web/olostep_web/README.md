# Olostep Web Loader

This loader fetches content from a URL using the [Olostep](https://www.olostep.com) API. The loader can be used in two modes: `scrape` and `search`.

## Setup

```bash
pip install requests
```

You also need an API key from [Olostep](https://www.olostep.com).

## Usage

To use the loader, you need to instantiate `OlostepWebReader` with your API key and the desired mode.

### Scrape Mode

This mode scrapes a single URL and returns the content.

```python
from llama_index.readers.web import OlostepWebReader
from llama_index.core import SummaryIndex

# Initialize the reader in scrape mode
reader = OlostepWebReader(api_key="YOUR_OLOSTEP_API_KEY", mode="scrape")

# Load data from a URL
documents = reader.load_data(url="https://en.wikipedia.org/wiki/Earth")

# You can also pass additional parameters to the API
documents_with_params = reader.load_data(
    url="https://en.wikipedia.org/wiki/Earth", params={"formats": ["markdown"]}
)

# Create a summary index
index = SummaryIndex.from_documents(documents)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is the essay about?")
print(response)
```

### Search Mode

This mode performs a Google search and returns the results.

```python
from llama_index.readers.web import OlostepWebReader
from llama_index.core import SummaryIndex

# Initialize the reader in search mode
reader = OlostepWebReader(api_key="YOUR_OLOSTEP_API_KEY", mode="search")

# Load data using a search query
documents = reader.load_data(query="What are the latest advancements in AI?")

# You can also pass additional parameters, for example, to specify the country for the search
documents_with_params = reader.load_data(
    query="What are the latest advancements in AI?", params={"country": "US"}
)

# The result is a JSON object with the search results
print(documents[0].text)
```
