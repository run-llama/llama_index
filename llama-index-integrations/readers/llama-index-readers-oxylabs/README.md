# LlamaIndex Oxylabs Reader Integration

Use Oxylabs Reader to load the data from search engines, e-commerce sites, travel platforms, and any other website.
For more information checkout out the [Oxylabs documentation](https://developers.oxylabs.io/scraper-apis/web-scraper-api).

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

[Set up](https://oxylabs.io/) your Oxylabs account and get the username and password

### Using OxylabsReader

```python
from llama_index.readers.oxylabs import OxylabsReader


reader = OxylabsReader(
    username="OXYLABS_USERNAME",
    password="OXYLABS_PASSWORD",
)

docs = reader.load_data(
    {
        "source": "google_search",
        "parse": True,
        "query": "Iphone 16",
        "geo_location": "Paris, France",
    }
)

print(docs)
```
