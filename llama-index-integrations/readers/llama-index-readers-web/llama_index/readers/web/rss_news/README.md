# RSS News Loader

This loader allows fetching text from an RSS feed. It uses the `feedparser` module
to fetch the feed and the `NewsArticleReader` to load each article.

## Usage

To use this loader, pass in an array of URLs of RSS feeds. It will download the pages referenced in each feed and
combine them:

```python
from llama_index.core.readers.web.rss_news import RSSNewsReader

urls = [
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://roelofjanelsinga.com/atom.xml",
]

RSSNewsReader = download_loader("RSSNewsReader")
reader = RSSNewsReader()

documents = reader.load_data(urls=urls)
```

Or OPML content:

```python
with open("./sample_rss_feeds.opml", "r") as f:
    documents = reader.load_data(opml=f.read())
```

We can also pass in args for the NewsArticleLoader which parses each article:

```python
documents = reader.load_data(urls=urls, nlp=True)
```
