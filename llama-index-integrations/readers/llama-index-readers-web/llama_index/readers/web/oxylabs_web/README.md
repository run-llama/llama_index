# Oxylabs Webpage Loader

Use Oxylabs Webpage Loader to load a webpage from any URL.

For more information checkout out the [Oxylabs documentation](https://developers.oxylabs.io/scraper-apis/web-scraper-api).

## Instructions for OxylabsReader

### Setup and Installation

Installation with `pip`

```shell
pip install llama-index-readers-web
```

Installation with `poetry`

```shell
poetry add llama-index-readers-web
```

Installation with `uv`

```shell
uv add llama-index-readers-web
```

### Get Oxylabs credentials

[Set up](https://oxylabs.io/) your Oxylabs account and get the username and password.

### Using OxylabsReader

```python
from llama_index.readers.web import OxylabsWebReader


reader = OxylabsWebReader(
    username="OXYLABS_USERNAME",
    password="OXYLABS_PASSWORD",
)

docs = reader.load_data(["https://ip.oxylabs.io"])

print(docs)
```
