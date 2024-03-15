# Async Website Loader

```bash
pip install llama-index-readers-web
```

This loader is an asynchronous web scraper that fetches the text from static websites by converting the HTML to text.

## Usage

To use this loader, you need to pass in an array of URLs.

```python
from llama_index.readers.web import AsyncWebPageReader

# for jupyter notebooks uncomment the following two lines of code:
# import nest_asyncio
# nest_asyncio.apply()

loader = AsyncWebPageReader()
documents = loader.load_data(urls=["https://google.com"])
```

### Issues Jupyter Notebooks asyncio

If you get a `RuntimeError: asyncio.run() cannot be called from a running event loop` you might be interested in this (solution here)[https://saturncloud.io/blog/asynciorun-cannot-be-called-from-a-running-event-loop-a-guide-for-data-scientists-using-jupyter-notebook/#option-3-use-nest_asyncio]

### Old Usage

use this syntax for earlier versions of llama_index where llama_hub loaders where loaded via separate download process:

```python
from llama_index import download_loader

AsyncWebPageReader = download_loader("AsyncWebPageReader")

loader = AsyncWebPageReader()
documents = loader.load_data(urls=["https://google.com"])
```
