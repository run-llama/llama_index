# LlamaIndex Readers Integration: MediaWiki

## Overview

The MediaWiki Reader loads pages from any [MediaWiki](https://www.mediawiki.org/)-based wiki (Wikipedia, Wikiversity, or your own instance) and returns them as LlamaIndex `Document` objects. It uses the wiki’s [Action API](https://www.mediawiki.org/wiki/API:Main_page) via [mwclient](https://github.com/mwclient/mwclient): you provide the wiki **host** (and optionally **path** and **scheme**), and the reader fetches page list, metadata (URL, last-modified), and parsed text, with optional namespace filtering.

### Features

- **Any MediaWiki instance** — Use `host`, `path`, and `scheme` to point at any wiki (e.g. `host="en.wikipedia.org"` with default path `/w/`).
- **Resource-based API** — Implements `load_resource` and `get_resource_info` for LlamaIndex ingestion and RAG pipelines.
- **Efficient listing** — Batched API calls for page metadata; optional `namespaces` filter.
- **HTML to text** — Converts wiki HTML to clean text via html2text (with regex fallback on failure).

### Installation

```bash
pip install llama-index-readers-mediawiki
```

The reader depends on **mwclient** for MediaWiki API access; it is installed automatically with the package.

### Usage

**Load a single page by title (get metadata with get_resource_info, then load content):**

```python
from llama_index.readers.mediawiki import MediaWikiReader

reader = MediaWikiReader(host="en.wikipedia.org")

title = "Python (programming language)"
info = reader.get_resource_info(title)
if info.get("url"):
    docs = reader.load_resource(
        title,
        resource_url=info["url"],
        last_modified=info.get("last_modified"),
    )
# docs is a list of one Document with .text and .metadata (title, url, last_modified)
```

**Stream all pages (lazy):**

```python
for doc in reader.lazy_load_data():
    print(doc.metadata.get("title"), len(doc.text))
```

**Optional: custom path/scheme, namespace filter, page limit:**

```python
reader = MediaWikiReader(
    host="mywiki.example.com",
    path="/w/",
    scheme="https",
    page_limit=100,
    namespaces=[0],  # Main namespace only
)
```

### Configuration

| Parameter    | Type                | Default       | Description                                                                                                                                                     |
| ------------ | ------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `host`       | `str`               | required      | MediaWiki site hostname (e.g. `en.wikipedia.org`).                                                                                                              |
| `path`       | `str`               | `"/w/"`       | MediaWiki script path (API at `{path}api.php`).                                                                                                                 |
| `scheme`     | `"https" \| "http"` | `"https"`     | URL scheme.                                                                                                                                                     |
| `page_limit` | `int`               | `500`         | Max page titles per allpages API call (pagination).                                                                                                             |
| `namespaces` | `list[int] \| None` | `None`        | Namespace IDs to list; `None` = wiki content namespaces from siteinfo API ([$wgContentNamespaces](https://www.mediawiki.org/wiki/Manual:$wgContentNamespaces)). |
| `logger`     | `logging.Logger`    | module logger | Logger instance (injectable for tests or custom logging). Not serialized.                                                                                       |

### Manual testing

For manual testing, run the usage examples above with your wiki’s `host` (and `path`/`scheme` if not default). For a **private wiki**, set `host`, call `reader.login(username, password)` (or use a bot password), then use `load_resource` / `lazy_load_data` as usual.

### License

MIT.

---

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index) and/or subsequently as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
