# AlterLab Web Reader

```bash
pip install llama-index-readers-web
```

[AlterLab](https://alterlab.io) is a web scraping API with anti-bot handling and JavaScript rendering. `AlterLabWebReader` loads pages as markdown `Document`s through the AlterLab REST API (`https://api.alterlab.io`, `X-API-Key` header). No extra SDK is required.

Set your API key in the `ALTERLAB_API_KEY` environment variable, or pass `api_key=`.

## Usage

```python
from llama_index.readers.web import AlterLabWebReader

# Scrape one page
reader = AlterLabWebReader(mode="scrape")
documents = reader.load_data(url="https://example.com")

# Crawl a site (extra request fields go in `params`)
reader = AlterLabWebReader(mode="crawl", params={"max_pages": 20, "max_depth": 2})
documents = reader.load_data(url="https://docs.example.com")

# Search the web
reader = AlterLabWebReader(mode="search", params={"num_results": 5})
documents = reader.load_data(query="web scraping api")
```

## Parameters

- `api_key`: AlterLab API key. Defaults to `ALTERLAB_API_KEY`.
- `mode`: `"scrape"` (default), `"crawl"` or `"search"`.
- `params`: extra fields for the API request, for example `{"mode": "js"}` to force JavaScript rendering when scraping. See the [API reference](https://alterlab.io/docs).
- `base_url`: API base URL. Defaults to `https://api.alterlab.io`.
- `poll_timeout`: maximum seconds to wait for a scrape or crawl job. Defaults to 300.

LLM-based extraction (`extraction_prompt` or `extraction_schema` in `params`) requires your own LLM provider key (OpenAI, Anthropic, OpenRouter and others) registered in your AlterLab account. Scraping, crawling and search do not.

A standalone package, [`llama-index-readers-alterlab`](https://pypi.org/project/llama-index-readers-alterlab/), is also available.
