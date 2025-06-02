# News Article Loader

```bash
pip install llama-index-readers-web
```

This loader makes use of the `newspaper3k` library to parse web page urls which have news
articles in them.

## Usage

```
pip install newspaper3k
```

Pass in an array of individual page URLs:

```python
from llama_index.readers.web import NewsArticleReader

reader = NewsArticleReader(use_nlp=False)
documents = reader.load_data(
    [
        "https://www.cnbc.com/2023/08/03/amazon-amzn-q2-earnings-report-2023.html",
        "https://www.theverge.com/2023/8/3/23818388/brave-search-image-video-results-privacy-index",
    ]
)
```
