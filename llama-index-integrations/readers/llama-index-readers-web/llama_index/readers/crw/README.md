# CRW reader

[CRW](https://github.com/us/crw) is a Firecrawl-compatible web scraper. This integration ships in the **`llama-index-readers-web`** package (no separate PyPI project).

## Install

```bash
pip install llama-index-readers-web
```

Run a CRW server (default API: `http://localhost:3000`).

## Usage

```python
from llama_index.readers.crw import CrwReader

reader = CrwReader()  # defaults to http://localhost:3000
docs = reader.load_data(url="https://example.com", mode="scrape")
```

- **`mode`**: `scrape` (single page), `crawl` (BFS crawl), or `map` (link discovery). Set on the reader or pass per call to `load_data`.

The same implementation is available as `CrwWebReader` under `llama_index.readers.web.crw_web`.
