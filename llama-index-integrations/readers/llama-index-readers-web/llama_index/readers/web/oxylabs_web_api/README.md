# Oxylabs Web API Loader

```bash
pip install llama-index-readers-web
```

`OxylabsWebAPIReader` loads web pages through the [Oxylabs Web API](https://oxylabs.io), including JavaScript-heavy and bot-protected pages, and returns them as Markdown (default) or HTML. It authenticates with a single API key: create a Web API instance in the [Oxylabs dashboard](https://dashboard.oxylabs.io) and set its key as `OXYLABS_WEB_API_KEY`. Keys from other Oxylabs products don't work.

For the Oxylabs Web Scraper API, which uses a username and password, see `OxylabsWebReader`.

## Usage

```python
from llama_index.readers.web import OxylabsWebAPIReader

# make sure the OXYLABS_WEB_API_KEY variable is set
reader = OxylabsWebAPIReader()

documents = reader.load_data(
    [
        "https://sandbox.oxylabs.io/products/1",
        "https://sandbox.oxylabs.io/products/2",
    ]
)

print(documents[0].metadata)  # source, final_url, status_code, request_id
print(documents[0].text[:250])
```

`aload_data` loads the urls concurrently, up to `max_concurrency` (default 5) at a time:

```python
documents = await reader.aload_data(urls)
```

## Options

| Option            | Default               | Purpose                                                                                            |
| :---------------- | :-------------------- | :------------------------------------------------------------------------------------------------- |
| `api_key`         | `OXYLABS_WEB_API_KEY` | Web API key                                                                                        |
| `output`          | `markdown`            | `markdown` or `html`                                                                               |
| `location`        | unset                 | Two-letter country code to fetch the pages from, e.g. `DE`                                         |
| `device`          | desktop               | `desktop` or `mobile`                                                                              |
| `run_js`          | `False`               | Render the pages' JavaScript. Use it for pages that render client-side and come back nearly empty. |
| `max_wait`        | `600`                 | Seconds to wait for a slow page that the API finishes in the background                            |
| `max_concurrency` | `5`                   | Pages `aload_data` fetches at the same time                                                        |

```python
reader = OxylabsWebAPIReader(location="DE", device="mobile", run_js=True)
```

## Errors

A url the API could not return (a faulted scrape, an unreachable site) is skipped with a warning naming the url and the reason, so the other urls still load. A rejected API key or a spent quota raises `OxylabsWebAPIError` (from `llama_index.readers.web.oxylabs_web_api.base`), since no url can succeed.

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
