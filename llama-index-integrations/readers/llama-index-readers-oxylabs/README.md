# LlamaIndex Oxylabs Reader Integration

Use Oxylabs Reader to load data from Google Search and Amazon.
To scrape any other website, use `OxylabsWebReader` from `llama-index-readers-web`.
For more information check out the [Oxylabs documentation](https://developers.oxylabs.io/products/web-scraper-api).

## Instructions for OxylabsReader

### Setup and Installation

Installation with `pip`

```shell
pip install llama-index-readers-oxylabs
```

Installation with `poetry`

```shell
poetry add llama-index-readers-oxylabs
```

Installation with `uv`

```shell
uv add llama-index-readers-oxylabs
```

### Get Oxylabs credentials

[Set up](https://oxylabs.io/) your Oxylabs account and get the username and password.

### Using OxylabsReader

```python
from llama_index.readers.oxylabs import OxylabsGoogleSearchReader


reader = OxylabsGoogleSearchReader(
    username="OXYLABS_USERNAME",
    password="OXYLABS_PASSWORD",
)

docs = reader.load_data(
    {"query": "Iphone 16", "parse": True, "geo_location": "Berlin, Germany"}
)

print(docs[0].text)
```
