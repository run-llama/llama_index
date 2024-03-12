# Sitemap Loader

```bash
pip install llama-index-readers-web
```

This loader is an asynchronous web scraper that fetches the text from static websites by using its sitemap and optionally converting the HTML to text.

It is based on the [Async Website Loader](https://llama-hub-ui.vercel.app/l/web-async_web)

## Usage

To use this loader, you just declare the sitemap.xml url like this:

```python
from llama_index.readers.web import SitemapReader

# for jupyter notebooks uncomment the following two lines of code:
# import nest_asyncio
# nest_asyncio.apply()

loader = SitemapReader()
documents = loader.load_data(
    sitemap_url="https://gpt-index.readthedocs.io/sitemap.xml"
)
```

Be sure that the sitemap_url contains a proper [Sitemap](https://www.sitemaps.org/protocol.html)

## Filter option

You can filter locations from the sitemap that are actually being crawled by adding the _filter_ argument to the load_data method

```python
documents = loader.load_data(
    sitemap_url="https://gpt-index.readthedocs.io/sitemap.xml",
    filter="https://gpt-index.readthedocs.io/en/latest/",
)
```

## Issues Jupyter Notebooks asyncio

If you get a `RuntimeError: asyncio.run() cannot be called from a running event loop` you might be interested in this (solution here)[https://saturncloud.io/blog/asynciorun-cannot-be-called-from-a-running-event-loop-a-guide-for-data-scientists-using-jupyter-notebook/#option-3-use-nest_asyncio]

### Old Usage

use this syntax for earlier versions of llama_index where llama_hub loaders where loaded via separate download process:

```python
from llama_index import download_loader

SitemapReader = download_loader("SitemapReader")

loader = SitemapReader()
documents = loader.load_data(
    sitemap_url="https://gpt-index.readthedocs.io/sitemap.xml"
)
```
