# Spider Web Reader

[Spider](https://spider.cloud/?ref=langchain) is the [fastest](https://github.com/spider-rs/spider/blob/main/benches/BENCHMARKS.md#benchmark-results) crawler. It converts any website into pure HTML, markdown, metadata or text while enabling you to crawl with custom actions using AI.

Spider allows you to use high performance proxies to prevent detection, caches AI actions, webhooks for crawling status, scheduled crawls etc...

## Prerequisites

You need to have a Spider api key to use this loader. You can get one on [spider.cloud](https://spider.cloud).

```pip
pip install llama-index
```

```python
# Scrape single URL
from llama_index.readers.web import SpiderWebReader

spider_reader = SpiderWebReader(
    api_key="YOUR_API_KEY",  # Get one at https://spider.cloud
    mode="scrape",
    # params={} # Optional parameters see more on https://spider.cloud/docs/api
)

documents = spider_reader.load_data(url="https://spider.cloud")
print(documents)
```

```python
# Crawl domain with deeper crawling following subpages
from llama_index.readers.web import SpiderWebReader

spider_reader = SpiderWebReader(
    api_key="YOUR_API_KEY",
    mode="crawl",
    # params={} # Optional parameters see more on https://spider.cloud/docs/api
)

documents = spider_reader.load_data(url="https://spider.cloud")
print(documents)
```

For guides and documentation, visit [Spider](https://spider.cloud/docs/api)
