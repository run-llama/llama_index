# RSS Loader

```bash
pip install llama-index-readers-web
```

This loader allows fetching text from an RSS feed. It uses the `feedparser` module
to fetch the feed and optionally the `html2text` module to sanitize it.
allow modify feedparser's useragent

## Usage

To use this loader, pass in an array of URL's.

```python
from llama_index.readers.web import RssReader

reader = RssReader()
documents = reader.load_data(
    [
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "https://roelofjanelsinga.com/atom.xml",
    ]
)
```
